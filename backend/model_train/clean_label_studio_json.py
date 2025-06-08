# %%
import json

# %%
with open('model_train/data/test_set/test_set_labels.json', 'r') as f:
    data = json.load(f)


clean_results = []
for item in data:
    if item['annotations'] and len(item['annotations']) > 0:
        clean_results.append({
            "pic_id": item["file_upload"]
        })
        first_annotation = item['annotations'][0]
        
        textarea_results = [
            {k: v for k, v in result["value"].items() if k != 'rotation'}
            for result in first_annotation['result']
            if result['type'] == 'textarea'
        ]

        clean_results[-1]["text_areas"] = textarea_results
# %%
for result in clean_results:
    text_areas = result["text_areas"]
    for i, area in enumerate(text_areas):
        if area.get("text")[0] == "matricula":
            prev_item = text_areas[i-1] if i > 0 else None
            next_item = text_areas[i+1] if i < len(text_areas)-1 else None
            print(f"\nFor pic {result['pic_id']}:")
            print(f"Previous item: {prev_item}")
            print(f"Next item: {next_item}")


# %%
with open('model_train/data/test_set/test_set_labels.json', 'w') as f:
    json.dump(clean_results, f)
# %%
