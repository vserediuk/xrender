import os

shader_dir = "shaders_spv"
header_file = "shaders.h"

def generate_shader_header():
    header_lines = [
        "#pragma once\n\n", 
        "#include <vector>\n", 
        "namespace shaders {\n"
    ]
    
    for shader_filename in os.listdir(shader_dir):
        shader_path = os.path.join(shader_dir, shader_filename)
        if os.path.isfile(shader_path):
            shader_name = os.path.splitext(shader_filename)[0].replace('.', '_')

            with open(shader_path, 'rb') as f:
                shader_data = f.read()

            if len(shader_data) >= 4:
                magic_number = int.from_bytes(shader_data[:4], byteorder='little')
                if magic_number != 0x07230203:
                    print(f"Warning: {shader_filename} is not a valid SPIR-V file (Magic Number mismatch).")
                    continue

            header_lines.append(f"    const std::vector<unsigned char> {shader_name} = {{\n")
            
            header_lines.append("        " + ", ".join(f"0x{shader_data[i]:02x}" for i in range(len(shader_data))) + "\n")
            
            header_lines.append(f"    }};\n\n")

    header_lines.append("}\n")
    
    with open(header_file, 'w', encoding='utf-8') as f:
        f.writelines(header_lines)
    print(f"Header file '{header_file}' has been generated.")

generate_shader_header()
