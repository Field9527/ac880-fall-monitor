# Additional data, 2026-10-17

All files and temporary downloads are on E:. Network downloads contain data only.

CAUCAFall v4: https://data.mendeley.com/datasets/7w7fccy7ky/4
DOI 10.17632/7w7fccy7ky.4. Authors: Jose Camilo Eraso, Elena Muñoz,
Mariela Muñoz, Jesus Pinto. License: CC BY 4.0. The author's Figure6 documents
YOLO classes 0=nofall, 1=fall. Four temporal samples per activity, ten activities
for each of ten subjects are selected. Original file SHA256 values are checked.
Subjects 1-7 are training, 8 is validation, 9-10 are final testing; an image
with only a class-0 person is a negative for the fallen-person detector.
Subject separation does not by itself prove different rooms or cameras.
The v6 training extension adds12 temporal positions for each of70 activities of
subjects1-7 only (840 additional RGB images plus author labels). Total public
RGB images1240:1120 training,40 validation,80 test. Original v5 validation/test
files remain unchanged. v5 test results were already inspected; v6 repeats are
reported as regressions, not a newly unseen final holdout.

Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
Zalando Research; Han Xiao, Kashif Rasul, Roland Vollgraf (2017). MIT license
included with the subset. First 2,000 official training images and first 1,000
official test images, retaining the official train/test boundary. Complete IDX
records are read from the official gzip stream; subset SHA256 is recorded and
no claim is made to have checked the unretained full-source gzip MD5.
These 28x28 grayscale garments are auxiliary object negatives only, not a
substitute for RGB fall/ADL scenes or a camera field-accuracy benchmark.

UR Fall: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
Bogdan Kwolek and Michal Kepski, Human fall detection on embedded platform
using depth maps and wireless accelerometer, CMPB 117(3), 2014, 489-501.
Author license: CC BY-NC-SA 4.0, non-commercial academic use. Two official RGB
MP4 samples and author posture CSVs have been fetched for inspection; they
are not yet used by the v5 training importer. Video-level fall labels must
not be copied to all frames: author posture 0 is a transition, -1 is not
lying, 1 is lying. Cross-stream frame alignment needs checking before use.

Hand detection and 21 landmarks: OpenCV Zoo commit48b10fa505aeb51fbd127fc60a5c660040e823b7, Apache-2.0; source hashes in models/gestures/source_manifest.json. FreiHAND is used for coarse hand-region research supervision under its research/noncommercial terms; no original images are packaged.
