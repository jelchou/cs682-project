
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time



# Loss function
criterion = nn.BCEWithLogitsLoss()

def discriminator_loss(real_output, fake_output):
    # Labels for real images are all ones, for fake images are all zeros
    real_labels = torch.ones_like(real_output)
    fake_labels = torch.zeros_like(fake_output)

    # Real images loss
    real_loss = criterion(real_output, real_labels)

    # Fake images loss
    fake_loss = criterion(fake_output, fake_labels)

    # Total loss
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    # Generator wants discriminator to think that fake images are real
    real_labels = torch.ones_like(fake_output)
    return criterion(fake_output, real_labels)

def train_step(generator, discriminator, images, noise_dim, optimizer_g, optimizer_d, criterion):
    print("train_step called")  # Confirming train_step is called
    batch_size = images.size(0)
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # Training Discriminator
    optimizer_d.zero_grad()
    
    # Real images
    # print("Processing real images")  # Before processing real images
    real_output = discriminator(images)
    d_loss_real = criterion(real_output, real_labels)

    # Fake images
    # noise = torch.randn(batch_size, noise_dim, 1, 1)
    noise = torch.randn(batch_size, noise_dim)
    # print("Noise Vector Shape:", noise.shape)
    # print("Generating fake images")  # Before generating fake images
    fake_images = generator(noise)
    # print("Generated Images Shape (before detach):", fake_images.shape)
    # fake_images_detached = fake_images.detach()
    # print("Generated Images Shape (after detach):", fake_images_detached.shape)
    fake_output = discriminator(fake_images.detach())
    d_loss_fake = criterion(fake_output, fake_labels)

    # Backprop and optimize
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_d.step()

    # Training Generator
    optimizer_g.zero_grad()
    
    # Getting discriminator's predictions on fake images
    output = discriminator(fake_images)
    g_loss = criterion(output, real_labels)

    # Backprop and optimize
    g_loss.backward()
    optimizer_g.step()

    return d_loss, g_loss

def generate_and_save_images(generator, epoch, test_input):
    with torch.no_grad():
        generator.eval()  # Set the generator to evaluation mode
        generated_images = generator(test_input).cpu()

    # Plot and save images
    fig = plt.figure(figsize=(4, 4))
    for i in range(generated_images.size(0)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i].permute(1, 2, 0) * 0.5 + 0.5)  # Rescale to [0, 1]
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def model_save(generator, discriminator, optimizer_g, optimizer_d, epoch, file_path='model_checkpoint'):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
    }, f'{file_path}_{epoch}.pth')


def train(generator, discriminator, dataset, epochs, noise_dim, optimizer_g, optimizer_d, criterion, fixed_noise):
    for epoch in range(epochs):
        start = time.time()

        for i, (images, _) in enumerate(dataset):
            d_loss, g_loss = train_step(generator, discriminator, images, noise_dim, optimizer_g, optimizer_d, criterion)

        generate_and_save_images(generator, epoch + 1, fixed_noise)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            model_save(generator, discriminator, optimizer_g, optimizer_d, epoch + 1)

    generate_and_save_images(generator, epochs, fixed_noise)