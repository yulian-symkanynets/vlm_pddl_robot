"""Generated policy. Solved 37/40 training scenes (round 3). Review before use."""

def policy(v):
    todo = [o for o in v.objects() if not v.done(o)]
    if not todo:
        return None
    clear = [o for o in todo if v.is_clear(o)]

    def home(o):
        """A final place for o that is available now, else None."""
        z = v.goal(o)
        if z is None:
            return None
        tops = [t for t in v.legal_destinations(o)
                if t in v.objects() and v.done(t) and v.region(t) == z]
        if not v.can_support(o) and tops:
            return tops[0]                    # an object that ends a stack goes on a stack
        if v.has_room(z):
            waiting = [x for x in todo if x != o and v.goal(x) == z and v.can_support(x)]
            if v.can_support(o) or not waiting or tops:
                return z
            return None                       # keep the ground pose for an object that can carry
        return tops[0] if tops else None

    # 1. something can go straight to its final place (objects that can carry first)
    for o in sorted(clear, key=lambda o: (not v.can_support(o), o)):
        d = home(o)
        if d is not None:
            return (o, d)

    # 2. otherwise park the clear object that uncovers something useful soonest
    def depth_to_useful(o):
        n, below = 1, v.on(o)
        while below in v.objects():
            if v.done(below):
                return n
            z = v.goal(below)
            if z is not None and v.has_room(z):
                return n
            n, below = n + 1, v.on(below)
        return n if below in v.zones() else 99

    parked = set(v.free_slots())
    movable = [o for o in clear if v.on(o) not in parked]
    for o in sorted(movable, key=lambda o: (depth_to_useful(o), o)):
        if v.region(o) not in v.zones() and v.on(o) not in v.objects():
            continue                          # alone on the table or in a slot: moving it helps nobody
        if v.free_slots():
            return (o, v.free_slots()[0])
        spare = [z for z in v.zones() if v.has_room(z) and z != v.region(o)]
        if spare:
            return (o, spare[0])
    # 3. last resort: any object with a goal into its zone
    for o in clear:
        if v.goal(o) is not None and v.has_room(v.goal(o)):
            return (o, v.goal(o))
    return None
