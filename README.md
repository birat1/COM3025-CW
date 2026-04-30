# Deep Learning and Advanced AI Coursework

## Indoor Assistive Vision System for Visually Impaired Users

**Aim**: to test whether the system can provide useful indoor scene awareness for visually impaired users.  
**Team members**: Manu Malakannavar, Yuken Rai, Birat Ale, Amulyaa Laulkar

## Backend

Run the app

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

post an image to test backend

```bash
curl -X POST "http://127.0.0.1:8000/analyse-frame" -F "file=@test_images/01.jpg"
```

images to test are in test_images folder

## Frontend