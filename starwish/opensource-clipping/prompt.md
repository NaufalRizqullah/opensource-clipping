# opensource-clipping

Render one video that the customer has permission to process. The source is an attached MP4 or a public YouTube video supplied in the structured parameter `youtube_url`.

Read one to three intervals from `params.ranges` in request order. Each interval has `start` and `end`, measured from the beginning of the source in seconds or timestamps such as `06:00` and `01:06:00.5`. Each end must follow its start. If ranges are omitted, use 0.2 to 59.6 seconds.

Center-crop each interval to 9:16 and encode to 720 × 1280. Keep the source audio when present. Deliver `clip.mp4` and `thumbnail.jpg` for the first interval, and append `_2` or `_3` before the extensions for additional intervals. Deliver `metadata.json` with absolute source times, measured durations, and filenames for all requested clips. Report success after verifying the complete set.
