# Your own photos for the live benchmark

Put a photo here and describe what is really in it:

    four_towers.jpg          the photo (jpg or png)
    four_towers.json         the TRUE scene, same format as a request's "scene" (objects with "on", zones)
    four_towers.tasks.json   optional: [{"id", "instruction", "reference"}]; "reference" is a goal formula
                             (goal_logic.py) that must be true at the end. Without this file the photo gets
                             the sort and the stack-by-colour task.

`four_towers.json` and `four_towers.tasks.json` are ready: save the photo of 20 November as
`live_photos/four_towers.jpg`. Then:

    python live_bench.py --suites photo --photos live_photos

The true scene is never sent to the model when the photo is the input; it is only used to judge.
The same task is also sent as JSON, so you see photo route against JSON route for your own scene.
