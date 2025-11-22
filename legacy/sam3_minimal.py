from transformers import AutoModel, AutoProcessor

model_id = "facebook/sam3-hiera-small"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
