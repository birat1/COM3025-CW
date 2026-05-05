# Deep Learning and Advanced AI Coursework

## Indoor Assistive Vision System for Visually Impaired Users

**Aim**: to test whether the system can provide useful indoor scene awareness for visually impaired users.  
**Team members**: Manu Malakannavar, Yuken Rai, Birat Ale, Amulyaa Laulkar

## Backend setup

Setup a virtual environment

```bash
python3 -m venv .venv 
source .venv/bin/activate
```

Install packages

```bash
pip install -r requirements.txt
```

Run the app

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

To test the backend, you can upload images on the docs: `http://0.0.0.0:8000/docs`

All the scripts used to compare and evaluate detection (YOLO, DETR) and captioning (BLIP, ViT-GPT2, GIT) models can be found in the `model_evaluation` folder.

## Frontend setup

Install dependencies

```js
npm install
```

Start application

```js
npx expo start
```

Ensure your phone and computer are connected to the same Wifi network.  
Use the Expo Go app to scan the QR code displayed in your terminal after running the frontend.  

Remember to create an .env file with the API URL in the frontend using the .env.example.

For Mac, the public API URL for the backend can be found by running `ipconfig getifaddr en0` on the terminal.
