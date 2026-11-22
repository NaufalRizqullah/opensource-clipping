# Vertical Video Clip

Render one video that the customer has permission to process. The source is an attached MP4 or a public YouTube video supplied in the structured parameter `youtube_url`.

Cut the source from 0.2 to 59.6 seconds, center-crop to 9:16 and encode to 720 × 1280. Keep audio when the source has an audio track. Deliver the video, first-frame JPEG and timing JSON in the workspace output folder. Report success only after all three files pass verification.

The Python harness implements this fixed operation. All jobs use the same interval, including jobs that supply a free-form description.
