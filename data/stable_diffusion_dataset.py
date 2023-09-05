
def dataset_laion(num):
    from datasets import load_dataset
    print(f"loading {num} dataset from HuggingFace")    
    if num == "2B":
        # 2B dataset
        dataset = load_dataset("laion/laion2B-en", split="train")         
    elif num == "400M":
        # 400m dataset
        dataset = load_dataset("laion/laion400m", split="train")
    return dataset

# uncomment this for testing the script
dataset_laion("2B") 