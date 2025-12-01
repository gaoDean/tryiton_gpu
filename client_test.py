import requests
import argparse
import sys
import os

def test_inference(url, person, upper=None, lower=None, dress=None, shoe=None, bag=None, output="output.png"):
    files = {}
    
    # Helper to open files
    open_files = []
    try:
        if not os.path.exists(person):
            print(f"Error: Person image '{person}' not found.")
            return

        files['person'] = open(person, 'rb')
        open_files.append(files['person'])

        garments = {'upper': upper, 'lower': lower, 'dress': dress, 'shoe': shoe, 'bag': bag}
        for key, path in garments.items():
            if path:
                if not os.path.exists(path):
                     print(f"Error: {key} image '{path}' not found.")
                     return
                f = open(path, 'rb')
                files[key] = f
                open_files.append(f)

        print(f"Sending request to {url}...")
        response = requests.post(url, files=files, data={'steps': 30, 'seed': 42})

        if response.status_code == 200:
            with open(output, 'wb') as f:
                f.write(response.content)
            print(f"Success! Result saved to {output}")
            print("\n--- Inference Statistics ---")
            for key, value in response.headers.items():
                if key.startswith("X-"):
                    print(f"{key[2:]}: {value}")
            print("----------------------------")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Exception: {e}")
    finally:
        for f in open_files:
            f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FastFit Server")
    parser.add_argument("--url", type=str, default="http://localhost:5000/prompt", help="Server URL")
    parser.add_argument("--person", type=str, required=True, help="Path to person image")
    parser.add_argument("--upper", type=str, help="Path to upper garment")
    parser.add_argument("--lower", type=str, help="Path to lower garment")
    parser.add_argument("--dress", type=str, help="Path to dress")
    parser.add_argument("--shoe", type=str, help="Path to shoe")
    parser.add_argument("--bag", type=str, help="Path to bag")
    parser.add_argument("--output", type=str, default="client_output.png", help="Output filename")

    args = parser.parse_args()
    
    test_inference(args.url, args.person, args.upper, args.lower, args.dress, args.shoe, args.bag, args.output)
