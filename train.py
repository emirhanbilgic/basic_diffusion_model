import torch.optim as optim

#constants
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = 64

#data
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = CustomDataset(csv_file='data.csv', img_dir='images', tokenizer=tokenizer, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#models
text_encoder = TextEncoder()
diffusion_model = DiffusionModel()

#loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(list(text_encoder.parameters()) + list(diffusion_model.parameters()), lr=LEARNING_RATE)

#training loop
for epoch in range(EPOCHS):
    for batch in dataloader:
        text_repr = text_encoder(batch['encoded_text'])
        generated_images = diffusion_model(text_repr)
        loss = criterion(generated_images, batch['image'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")
