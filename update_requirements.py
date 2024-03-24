import subprocess

# Run pip freeze and capture the output
output = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE, text=True).stdout

# Filter out lines with "@ file://"
filtered_packages = [line for line in output.splitlines() if "@ file://" not in line]

# Save the filtered list to requirements.txt
with open("requirements.txt", "w") as f:
    f.write("\n".join(filtered_packages))
