## KERAS-DCGAN ##

Fork of [jacobgil/keras-dcgan](https://github.com/jacobgil/keras-dcgan)

---

## Usage

### Training

```
python3 dcgan.py train <args...>
```

### Image generation

(TODO)

---

## Utilities

### Resize image

```
(IMAGES_DIR)$ python (PATH)/utils/resize-images.py <DST_DIR> [--width WIDTH] [--height HEIGHT]
```

### Flip image (Horizontal)

```
(IMAGES_DIR)$ python (PATH)/utils/flip-images.py
```

### Sync folers

```
$ python (PATH)/utils/sync-dir.py <SOURCE_DIR> <TARGET_DIR>
```

- Files in the target dir will be removed if no file name matched in the source dir.

---

## Reference

[jacobgil/keras-dcgan](https://github.com/jacobgil/keras-dcgan)

[はじめてのGAN](https://elix-tech.github.io/ja/2017/02/06/gan.html)

[自前のデータでDCGANをやってみる - Qiita](https://qiita.com/nabechi6011/items/95eeb1d8aec2598efc65)

[Keras 2 で”はじめてのGAN” - Qiita](https://qiita.com/IntenF/items/94da17a8931e1f14b6e3)
