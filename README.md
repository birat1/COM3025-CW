# Deep Learning and Advanced AI Coursework

## Indoor Assistive Vision System for Visually Impaired Users

**Aim**: to test whether the system can provide useful indoor scene awareness for visually impaired users.  
**Team members**: Manu Malakannavar, Yuken Rai, Birat Ale, Amulyaa Laulkar

## Backend setup

Run the app

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

post an image to test backend

```bash
curl -X POST "http://127.0.0.1:8000/analyse-frame" -F "file=@test_images/01.jpg"
```

images to test are in test_images folder

## Frontend setup

Install dependencies
```
npm install
```

Start application
```
npx expo start
```

Ensure your phone and computer are connected to the same Wifi network

Use the Expo Go app to scan the QR code displayed in your terminal after running the start command
