# Promo video

How the YouTube promo is made: the same few fragments of a public-domain interview, played once in the original and
then dubbed into every target language. `scripts/promo_video.py` builds the video; the pipeline in the container
produces the audio tracks it reads.

The result looks like this: title card → *fragment 1* in English, Polish, French, German → *fragment 2* the same →
*fragment 3* the same → closing card. Every part has a `PART n/N` badge and a language label, so a viewer sees at a
glance that the voices are cloned, not re-recorded.

## 1. Pick a public-domain source

Anything that is free to reuse works; NASA material is public domain and easy to script:

* the SVS feed used by the E2E test: `https://svs.gsfc.nasa.gov/vis/a010000/a013800/a013868/ISS_HIGH_DEF_FEED.mp4`
  (19 minutes, a Hubble live interview),
* the NASA Image and Video Library: `https://images-api.nasa.gov/search?q=astronaut%20interview&media_type=video`,
  then `https://images-api.nasa.gov/asset/<nasa_id>` for the `~orig.mp4` / `~medium.mp4` files and a `.srt`.

Choose material with **clear, continuous speech**: cut a candidate window and transcribe it before committing to it
(`python` with `faster_whisper`, or the container). NASA feeds often open with minutes of music — the first minutes of
the Demo-2 briefing, for example, transcribe as `"Thank you."` or `"🎵"`, which is Whisper hallucinating on music.

## 2. One speaker per fragment

The point of the promo is that each fragment is a different voice. Check this *before* dubbing, because the pipeline
will happily dub the same person three times:

* run the pipeline once on a sampler (six 20 s windows from different offsets, concatenated) and read
  `debug_*/speakers.json` plus the `speaker` column of `script_initial.json`,
* a single interview usually has two speakers (host + guest). A 19-minute feed gave exactly two, so the third
  fragment came from a second NASA video,
* to tell a male from a female voice without listening, `core/gender_classifier.py` can be called directly on a wav:

  ```python
  from core.gender_classifier import GenderClassifier
  GenderClassifier().classify([samples_16k])  # ('male', 0.99)
  ```

  It is trained on speech, so a music-only window gives nonsense — always pair it with a transcription.

## 3. Assemble one source clip

Cut the chosen windows and concatenate them with a short silence between them, so the pipeline sees one file and the
promo can cut all fragments from the same offsets:

```bash
ffmpeg -v error -ss 8.9  -t 13.1 -i feed.mp4 -vf "scale=1280:720,fps=30,format=yuv420p" \
  -c:v libx264 -preset medium -crf 20 -c:a aac -b:a 192k -ar 48000 -ac 2 -y frag/a.mp4
# ... the same for frag/b.mp4 and frag/c.mp4, plus a 0.5 s black frag/gap.mp4
ffmpeg -v error -i frag/a.mp4 -i frag/gap.mp4 -i frag/b.mp4 -i frag/gap.mp4 -i frag/c.mp4 \
  -filter_complex "[0:v][0:a][1:v][1:a][2:v][2:a][3:v][3:a][4:v][4:a]concat=n=5:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -r 30 \
  -c:a aac -b:a 192k -ar 48000 -ac 2 -y promo-src.mp4
```

Use the `concat` **filter**, not `concat -c copy`: a stream copy of parts with slightly different audio extradata
silently drops the audio of the later parts (the third fragment came out at −51 dB).

## 4. Dub it

One run for all languages — the promo needs every track in the same file:

```bash
docker run -d --name dubarr-promo --gpus all --network host \
  -e PUID=1002 -e PGID=1002 -e HF_TOKEN -e TARGET_LANGS=pl,fr,de -e DEBUG=1 \
  -e LLM_API_URL=http://127.0.0.1:8080/v1 -e LLM_API_MODEL=gemma \
  -v "$PWD/work:/work" -v "$PWD/work/appdata:/app/data" -v "$PWD/work/promo/videos:/app/videos" \
  dubarr:e2e python3 /tmp/full_run.py "/work/promo/input/promo-src.mp4"
```

A 45 s source (4 lines) takes about 10 minutes with `TAKE_CHECK=judge`; the log ends with one
`Production report (<lang>)` per language, and all lines should be `ACCEPTED`.

## 5. Build the video

```bash
python scripts/promo_video.py promo-src.mp4 promo.mp4 \
  --windows 0-13.1,13.6-29.0,29.5-45.5 --langs pl,fr,de --thumbnail promo.jpg
```

The windows are the fragment offsets in `promo-src.mp4` (the gaps included), so they must match what you concatenated
in step 3. The script writes 1280×720 H.264/AAC with `+faststart`, which YouTube accepts as is.

## 6. Check the result

Transcribe a few seconds of each part and confirm the language and the speaker; a mis-labelled track is the one
mistake that survives to the upload:

```bash
ffmpeg -v error -ss 20 -t 7 -i promo.mp4 -ac 1 -ar 16000 -y /tmp/check.wav
# whisper: expect "pl" for the Polish part, "fr" for the French one, ...
```

Short slices (under ~10 s) can be mis-detected when the dub pronounces English proper names phonetically — use 15–20 s
before concluding that a track is wrong.

## 7. Publish

Upload the MP4, set the thumbnail, and link it from the README (`## 🎬 Demo`). Keep the CI-generated demo video in
place: the E2E workflow republishes it to the `demo` release on every push to `main`, so the README shows both the
fresh, low-quality CPU run and this GPU promo.

## Gotchas

* **`-map` disables the automatic stream selection.** Mapping only the audio silently produces an audio-only part,
  and the concatenation of those with the cards is what a player shows as a corrupt stream.
* **The original track is tagged `und`.** The mux tags only the dubbed tracks, so `scripts/promo_video.py` picks the
  original as "the last track that is not one of the target languages", not by `eng`.
* **`concat -c copy` is not safe** across parts encoded separately (see step 3).
* **Music is not silence.** Gender and language detection both produce confident nonsense on it; always check the
  transcription of a window first.
