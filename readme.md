This is a fork of [groinge/sd-webui-untitledmerger](https://github.com/groinge/sd-webui-untitledmerger) which I decided to maintain while the original maintainer is unavailable for personal reasons.
Within paretheses below is my comments as I want to maintain the original repos README.md as much as possible.

Capable of reusing calculations from recent merges to greatly improve merge times

Note:
- Only supports safetensors.
- Developed for webui 1.7.0 dev branch(But has been confirmed to work on 1.9.3 after I removed the image generation section temporarily)
- I'm currently focusing on 1.5 merging. XL merging generally works but there may be issues. (I have had no direct issues with SDXL so far)

Inspired by and uses snippets from: https://github.com/hako-mikan/sd-webui-supermerger 🙏🙏🙏

Power-Up merge from:
- https://github.com/martyn/safetensors-merge-supermario
- https://github.com/yule-BUAA/MergeLM

To do:
- XYZ plotting
- State saving
- History
(- Add back improved image generator)
(- Improve key matching and add per key weight settings menu)
(- Redo block weights to match format used by Supermerger to allow easy conversion of presets)
(- Merge history logger for basic information)