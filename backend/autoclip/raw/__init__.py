"""Stream-clips pipeline: long stream VOD in, ranked 9:16 clips out.

Each clip is reframed to 9:16 around the speaker, with burned-in captions,
ready to post. A Premiere timeline (``timeline.xml``) of the clips and a
manifest come with them.

Reuses upstream's transcription, highlight provider, boundary refinement,
reframe, and caption export. Everything lives in one output folder, and every
stage is skipped when its artifact already exists, so a failed run resumes.
"""
