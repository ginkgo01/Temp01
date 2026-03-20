from datasets import load_from_disk

ds = load_from_disk("math500_hf")    

print(ds["test"][306])