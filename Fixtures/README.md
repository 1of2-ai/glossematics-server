# Core ML test fixtures

`JinaV5OmniSmall.w8a16.dummy.bundle` is the checked-in, compiled Core ML golden fixture.
It has the production schema and assets but deterministic constant outputs. Server tests use
it without loading the full model or depending on GlossematicsCoreML at test time.

`golden-video.mp4` is a generated, one-second H.264 clip with eight distinct 64×64 frames.
It verifies the AVFoundation file and uploaded-byte paths. It can be reproduced with:

```bash
ffmpeg -f lavfi -i 'testsrc=size=64x64:rate=8:duration=1' \
  -c:v libx264 -preset ultrafast -crf 25 -pix_fmt yuv420p golden-video.mp4
```

`golden-long-video.mp4` is a 20-second, 10 fps, 256×256 H.264 clip. The bounded serving
profile samples 32 frames and reaches the converted video tower's largest f2048 patch bucket:

```bash
ffmpeg -f lavfi -i 'testsrc=size=256x256:rate=10:duration=20' \
  -an -c:v libx264 -preset veryfast -crf 34 -pix_fmt yuv420p \
  -movflags +faststart golden-long-video.mp4
```

The video files are test inputs only. `ffmpeg` is not required to build or test the daemon.
