"""Generated heuristic (candidate 2). Review before use."""

def h(v):
    # objects that still have to move at least once
    return sum(1 for o in v.objects() if not v.done(o))
