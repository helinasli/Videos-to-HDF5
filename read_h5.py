import h5py
import argparse

# file_path = './out/out.h5'
# output_file = './test_video_h5_file_output.txt'  # Specify the path for the output file

def read_h5(file_path, output_file):
    # Open the output file in write mode
    with open(output_file, 'w') as out_file:
        # Open the H5 file in read mode
        with h5py.File(file_path, 'r') as h5_file:
            # Iterate through keys to display the structure and content of the file
            for key in h5_file.keys():
                out_file.write(f"Key: {key}\n")
                for sub_key in h5_file[key]:
                    data = h5_file[key][sub_key][()]
                    out_file.write(f"  /{sub_key}:\n")

                    # Check if the data is an array or a scalar
                    if isinstance(data, (list, tuple)) or hasattr(data, 'shape'):
                        out_file.write(f"    Shape: {data.shape}, Data type: {data.dtype}\n")
                        out_file.write("    Content:\n")
                        out_file.write(f"{data}\n")
                    else:
                        # Handle scalar or non-array data
                        out_file.write(f"    Value: {data}, Data type: {type(data)}\n")

                    out_file.write("\n")  # Add a blank line for better readability

    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="File to read")
    parser.add_argument("--output_file", type=str, required=True, help="The path of the output")
    args = parser.parse_args()

    read_h5(args.file_path, args.output_file)