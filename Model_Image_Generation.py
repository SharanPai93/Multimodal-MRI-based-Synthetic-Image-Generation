def generate_images(model,scheduler,num_imgs, dims):
    # Generate initial random noise on GPU
    noise = torch.randn((num_imgs, 1, dims[0], dims[1]), device="cuda")
    model.eval()

    with torch.no_grad():
        # Use scheduler timesteps (assumes scheduler.timesteps is defined)
        for t in scheduler.timesteps:
            t_tensor = torch.full((num_imgs,), t, device="cuda", dtype=torch.long)
            noise_pred = model(noise, t_tensor).sample
            noise = scheduler.step(noise_pred, t, noise).prev_sample

    # Print noise range for debugging
    print("Noise range before scaling:", noise.min().item(), noise.max().item())

    # Rescale from [-1,1] to [0,255]
    generated_images = (noise * 0.5 + 0.5) * 255
    generated_images = torch.clamp(generated_images, 0, 255).cpu().to(dtype=torch.uint8).numpy()
    print("Generated image pixel range:", generated_images.min(), generated_images.max())

    return generated_images

def save_images(generated_images,num_imgs,save_dir,add=0):
    os.makedirs(save_dir, exist_ok=True)
    # Save generated images
    for i in range(num_imgs):
        # Convert to PIL Image (mode "L" for grayscale)
        img = Image.fromarray(generated_images[i, 0], mode="L")
        # (Optional) Adjust brightness/contrast here if desired using ImageEnhance
        img.save(os.path.join(save_dir, f"image_{add + i + 1}.png"))

def show_images(generated_images,num_rows,num_cols,figsize,num_imgs):
    # Display images in a grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < num_imgs:
            ax.imshow(Image.fromarray(generated_images[i, 0], mode="L"), cmap="gray")
            ax.axis("off")
    plt.show()
