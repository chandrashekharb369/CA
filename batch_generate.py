import os
import time
from generate_dataset import generate_dataset

print("======================================================")
print("  Batch Generator - CA Intelligence Suite")
print("======================================================")

profiles = ["generic", "trading", "manufacturing", "service", "startup"]
num_rows = 12000

for profile in profiles:
    output_filename = os.path.join("test_dataset", f"dataset_{profile}.csv")
    print(f"\n🚀 Generating dataset for profile '{profile}' ({num_rows} rows)...")
    
    start_time = time.time()
    # It writes automatically internally, we just tell it the path if supported.
    # Looking at generate_dataset source, it writes to `output_path`.
    df = generate_dataset(n=num_rows, output_path=output_filename, company_type=profile)
    df.to_csv(output_filename, index=False)
    
    elapsed = time.time() - start_time
    print(f"✅ Saved to {output_filename} (took {elapsed:.1f}s)")

print("\n🎉 All 5 datasets generated successfully!")
