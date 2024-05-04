import os

def concatenate_python_files(source_directory, output_file):
    """
    Concatenates all Python files in the specified directory into a single file.

    Args:
    source_directory (str): The path to the directory containing Python files.
    output_file (str): The path to the output file where the concatenated result will be stored.
    """
    # Ensure the directory exists
    if not os.path.exists(source_directory):
        print(f"Error: The directory {source_directory} does not exist.")
        return

    # Create or overwrite the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Walk through the directory
        for dirpath, dirnames, filenames in os.walk(source_directory):
            for filename in filenames:
                if filename.endswith('.py'):
                    # Path to the current file
                    file_path = os.path.join(dirpath, filename)
                    # Writing the name of the file as a comment
                    outfile.write(f"\n# Start of {filename}\n")
                    # Open and read the current file
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                    # Writing the end of the file as a comment
                    outfile.write(f"\n# End of {filename}\n")

# Example usage
if __name__ == "__main__":
    source_dir = "./"  # Change to the path of your source directory
    output_path = "./concatenate.txt"  # Change to your desired output file path
    concatenate_python_files(source_dir, output_path)
