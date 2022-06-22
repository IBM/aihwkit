from src.aihwkit.nn.modules.transformer.transformer import AnalogBertModel

model = AnalogBertModel.from_pretrained("bert-base-uncased")

print(model)
