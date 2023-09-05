# facing dependency issues from my end 
def dataset_laion(num):
    from datasets import load_dataset    
    if num == "2B":
        # 2B dataset
        print("loading 2B dataset from HuggingFace")
        dataset = load_dataset("laion/laion2B-en", split="train")         
    elif num == "400M":
        # 400m dataset
        print("loading 400M dataset from HuggingFace")
        dataset = load_dataset("laion/laion400m", split="train")
    return dataset

# uncooment this for testing the script
# dataset_laion("2B") 