import torchfile
import ast

model = torchfile.load("../pre_trained_models/vrn-unguided.t7")

assert model is not None

print("vars:", list(vars(model)))
print("_obj:", model._obj.keys())

print("modules:")
for i, m in enumerate(model._obj[b'modules'], start=1):
    print("\t", i, "->", m._obj.keys())
