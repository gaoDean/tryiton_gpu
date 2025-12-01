import os
import sys
import tempfile
import shutil
import random
from flask import Flask, request, send_file, jsonify, make_response
from inference import FastFitEngine

app = Flask(__name__)

# Initialize engine globally
print("Initializing FastFit Engine...")
try:
    engine = FastFitEngine()
except Exception as e:
    print(f"Failed to initialize engine: {e}")
    sys.exit(1)

@app.route('/prompt', methods=['POST'])
def prompt():
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Save uploaded files
        files = {}
        required_files = ['person']
        optional_files = ['upper', 'lower', 'dress', 'shoe', 'bag']
        
        # Check required
        if 'person' not in request.files:
            return jsonify({"error": "Missing 'person' image"}), 400
            
        # Save all files
        for key in required_files + optional_files:
            if key in request.files:
                file = request.files[key]
                if file.filename == '':
                    continue
                path = os.path.join(temp_dir, f"{key}.jpg")
                file.save(path)
                files[key] = path

        if not files.get('person'):
             return jsonify({"error": "Missing person image content"}), 400

        # 2. Construct garments dict
        garments = {
            "upper": files.get('upper'),
            "lower": files.get('lower'),
            "dress": files.get('dress'),
            "shoe": files.get('shoe'),
            "bag": files.get('bag')
        }

        # Validate garment logic (simple check matching inference.py)
        if garments['dress'] and (garments['upper'] or garments['lower']):
            return jsonify({"error": "Cannot mix dress with upper/lower"}), 400
            
        if not any([garments['upper'], garments['lower'], garments['dress'], garments['shoe'], garments['bag']]):
             return jsonify({"error": "No garments provided"}), 400

        # 3. Run Inference
        # We ask engine to return the PIL image directly
        steps = int(request.form.get('steps', 30))
        seed_param = request.form.get('seed')
        if seed_param is None:
            seed = random.randint(0, 2**32 - 1) # Randomize seed if not provided
        else:
            seed = int(seed_param)
        
        print(f"Processing request in {temp_dir}")
        result_image, stats = engine.process(
            person_path=files['person'],
            garments=garments,
            output_path=None, # Return PIL Image
            steps=steps,
            seed=seed
        )
        
        # 4. Save result to temp to send it back
        output_path = os.path.join(temp_dir, "result.png")
        result_image.save(output_path)
        
        response = make_response(send_file(output_path, mimetype='image/png'))
        response.headers['X-Encoding-Time'] = stats.get('encoding_time', 'N/A')
        response.headers['X-Inference-Time'] = stats.get('inference_time', 'N/A')
        response.headers['X-Total-Time'] = stats.get('total_time', 'N/A')
        if 'max_vram' in stats:
            response.headers['X-Max-VRAM'] = stats['max_vram']
            
        return response

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
