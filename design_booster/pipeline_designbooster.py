from typing import Union, List, Optional, Callable, Dict, Any
import PIL
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import preprocess, StableDiffusionPipelineOutput


class StableDiffusionDesignBoosterPipeline(StableDiffusionPipeline):
    def _prepare_image(self, image, dtype, device, generator=None):
        assert not isinstance(generator, list), "Not Support!"
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )
        image = image.to(device=device, dtype=dtype)
        latents = self.vae.encode(image).latent_dist.sample(generator)
        latents = self.vae.config.scaling_factor * latents
        latents = torch.cat([latents], dim=0)
        return latents

    def _encode(self, prompt, image, batch_size, num_images_per_prompt, device, do_classifier_free_guidance, negative_prompt=None, generator=None):
        """
        prepare 2
        """
        batch_size = batch_size * num_images_per_prompt
        # proceed with batch_size=1 first
        latents = self._prepare_image(image, self.text_encoder.dtype, device, generator)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        text_embeds = self.text_encoder.text_model(input_ids=text_input_ids.to(device)).last_hidden_state
        image_embeds = self.text_encoder.image_model(latents)
        empty_image_embeds = torch.zeros(
                (1, self.text_encoder.config.num_image_prompt_tokens, self.text_encoder.config.projection_dim),
                device=image_embeds.device
        )
        zt_all = self.text_encoder.transformer(torch.cat((text_embeds, image_embeds), dim=1))
        zt_empty = self.text_encoder.transformer(torch.cat((text_embeds, empty_image_embeds), dim=1))

        bs_embed, seq_len, _ = zt_all.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        zt_all = zt_all.repeat(1, num_images_per_prompt, 1)
        zt_all = zt_all.view(bs_embed * num_images_per_prompt, seq_len, -1)
        zt_empty = zt_empty.repeat(1, num_images_per_prompt, 1)
        zt_empty = zt_empty.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # prepare
        if do_classifier_free_guidance:
            uncond_input = self.tokenizer(
                "" if negative_prompt is None else negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeds = self.text_encoder.text_model(
                input_ids=uncond_input.input_ids.to(device),
            ).last_hidden_state
            uncond_zt_all = self.text_encoder.transformer(torch.cat((uncond_embeds, image_embeds), dim=1))
            uncond_zt_empty = self.text_encoder.transformer(torch.cat((uncond_embeds, empty_image_embeds), dim=1))
            uncond_zt_all = uncond_zt_all.repeat(1, num_images_per_prompt, 1)
            uncond_zt_all = uncond_zt_all.view(bs_embed * num_images_per_prompt, seq_len, -1)
            uncond_zt_empty = uncond_zt_empty.repeat(1, num_images_per_prompt, 1)
            uncond_zt_empty = uncond_zt_empty.view(bs_embed * num_images_per_prompt, seq_len, -1)

            zt_all = torch.cat([uncond_zt_all, zt_all])
            zt_empty = torch.cat([uncond_zt_empty, zt_empty])
        return zt_all, zt_empty

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            image: Union[torch.FloatTensor, PIL.Image.Image] = None,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: Optional[float] = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            ######################
            sigma_switch_step: int = 0,
            **kwargs,
    ):
        assert prompt_embeds is None and negative_prompt_embeds is None, "Not support!"
        assert isinstance(prompt, str), "Not support!"
        assert 0 < sigma_switch_step <= num_inference_steps, f"`sigma_switch_step` must be between 0 and num_inference_steps={num_inference_steps}"
        batch_size = 1
        print(f"sigma_switch_step: {sigma_switch_step}")
        # preprocess image
        image = preprocess(image)
        height, width = image.shape[-2:]
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        device = self._execution_device
        # make sure to set eval to avoid dropout
        self.text_encoder.eval()
        do_classifier_free_guidance = guidance_scale > 1.0
        zt_all, zt_empy = self._encode(prompt, image, batch_size, num_images_per_prompt, device, do_classifier_free_guidance, negative_prompt, generator)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            zt_all.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=zt_all if t.item() >= sigma_switch_step else zt_empy,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, zt_all.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, zt_all.dtype)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
