

### 1\. Global Model (global-256.pt)

This model is used for the global context or first-stage processing.

  * **Filename**: `global-256.pt`
  * **Download Link**: [Download from Google Drive](https://www.google.com/url?sa=E&source=gmail&q=https://drive.google.com/uc?export=download%26id=1axJMm0fpg0v2HIApxz0IIaY0WTDKalCH)

You can download it directly using the command line:

```bash
wget -O checkpoints/global-256.pt "[https://drive.google.com/uc?export=download&id=1axJMm0fpg0v2HIApxz0IIaY0WTDKalCH](https://drive.google.com/uc?export=download&id=1axJMm0fpg0v2HIApxz0IIaY0WTDKalCH)"
```

### 2\. Defect Patch Model (defect-patch-64.pt)

This model is used for fine-grained defect analysis on patches.

  * **Filename**: `defect-patch-64.pt`
  * **Download Link**: [Download from Google Drive](https://www.google.com/url?sa=E&source=gmail&q=https://drive.google.com/uc?export=download%26id=1GHT-q1XjF_aCmInp_g5gMruX3yih-iyM)

You can download it directly using the command line:

```bash
wget -O checkpoints/defect-patch-64.pt "[https://drive.google.com/uc?export=download&id=1GHT-q1XjF_aCmInp_g5gMruX3yih-iyM](https://drive.google.com/uc?export=download&id=1GHT-q1XjF_aCmInp_g5gMruX3yih-iyM)"
```

After downloading, your directory structure should look like this:

```
├── checkpoints/
│   ├── global-256.pt
│   └── defect-patch-64.pt
├── src/
│   └── ...
└── README.md
```
