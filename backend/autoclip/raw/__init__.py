"""Raw-clips pipeline: long stream VOD in, ranked clips out in two versions.

- ``clips/``    raw 16:9, uncropped, uncaptioned, with handles (for Premiere)
- ``vertical/`` 9:16, speaker-tracked, burned-in captions (ready to post)

Reuses upstream's transcription, highlight provider, boundary refinement,
reframe, and caption export. Everything lives in one output folder, and every
stage is skipped when its artifact already exists, so a failed run resumes.
"""
