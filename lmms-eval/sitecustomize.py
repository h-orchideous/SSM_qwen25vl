try:
    import av

    if not hasattr(av, "AVError"):
        av.AVError = getattr(av, "FFmpegError", OSError)
except Exception:
    pass
