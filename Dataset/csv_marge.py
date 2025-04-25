import pandas as pd

# Step 1: Load the data
image_paths_df = pd.read_csv("/content/train_image_paths.csv")
labeled_studies_df = pd.read_csv("/content/train_labeled_studies.csv")

# Step 2: Rename columns for clarity
image_paths_df.columns = ["image_path"]
labeled_studies_df.columns = ["study_path", "label"]

# Step 3: Extract study path from each image path
image_paths_df["study_path"] = image_paths_df["image_path"].apply(lambda x: "/".join(x.strip().split("/")[:-1]))

# Step 4: Clean and normalize paths
labeled_studies_df["study_path"] = labeled_studies_df["study_path"].str.strip().str.rstrip("/")

# Step 5: Merge the label onto the image paths
merged_df = image_paths_df.merge(labeled_studies_df, on="study_path", how="left")

# Step 6: Drop the intermediate study_path column
final_df = merged_df.drop(columns=["study_path"])

# Step 7: Save the result to CSV without index
final_df.to_csv("merged_image_labels.csv", index=False)

print("✅ Merged file saved as 'merged_image_labels.csv'")
