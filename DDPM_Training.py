# Training loop for DDPM

earlyStopped = False # If using early stopping mechanism
best_checkpoint_epoch = None  # Track the best checkpoint epoch
check = True

for epoch in range(start_epoch, num_epochs+1):
    total_loss = 0  # Track total loss per epoch
    model.train()

    for real_images, _ in dataloader:
        real_images = real_images.to(device)

        # Generate random noise
        noise = torch.randn_like(real_images).to(device)

        # Pick random timesteps
        timesteps = torch.randint(0, 1000, (real_images.shape[0],), device=device).long()

        # Add noise to images
        noisy_images = scheduler.add_noise(real_images, noise, timesteps)

        # Predict noise using UNet model
        predicted_noise = model(noisy_images, timesteps).sample

        # Compute loss
        loss = loss_fn(predicted_noise, noise)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # Accumulate loss

    avg_loss = total_loss / len(dataloader)  # Compute average loss

    # Learning rate scheduling
    scheduler_lr.step(avg_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Current LR = {optimizer.param_groups[0]['lr']}")


    if check and (epoch - start_epoch) % 20 == 0:
        if epoch > 0: save_checkpoint(model, optimizer, epoch)
        generated_images = generate_images(model,scheduler,4,(128,128))
        show_images(generated_images,1,4,(8,2),4)

    # Check early stopping condition (Optional, comment out if needed)
    stop_training, is_best = early_stopping(avg_loss, epoch)
    
    if stop_training:
        earlyStopped = True
        break  # Stop training

    # Prevent saving best checkpoint when epoch is less than 20 (loss is unpredictable)
    if epoch >= 20:
        if is_best:
            best_checkpoint_epoch = epoch
            save_checkpoint(model, optimizer, epoch, best=True)  # Save best checkpoint
        elif (epoch - start_epoch) % 10 == 0:
            save_checkpoint(model, optimizer, epoch)  # Save every 10 epochs

    # Save checkpoint every 10 epochs
    elif (epoch - start_epoch) % 10 == 0:
        save_checkpoint(model, optimizer, epoch)


# Final checkpoint save
if not earlyStopped:
    save_checkpoint(model, optimizer, epoch)

print(f"Best model checkpoint saved at epoch {best_checkpoint_epoch}")
