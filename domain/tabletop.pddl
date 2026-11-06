; Tabletop manipulation domain for the Lite 6 + vacuum gripper.
; Written ONCE. Every scene/task only produces a new PROBLEM file.
; Plain STRIPS + typing, so every classical planner (Fast Downward, pyperplan) can use it.
;
; Supports of an item (exactly one is true at any time):
;   (on-table ?x)      free spot on the table (the local controller picks the coordinates)
;   (on ?x ?y)         resting on top of another item
;   (in ?x ?p)         inside / on a place (colored paper zone, box, shelf level) using one slot
;   (holding ?x)       in the gripper
(define (domain tabletop)
  (:requirements :strips :typing)
  (:types item place slot)

  (:predicates
    (on-table ?x - item)
    (on ?x - item ?y - item)
    (in ?x - item ?p - place)
    (occupies ?x - item ?s - slot)
    (slot-of ?s - slot ?p - place)
    (slot-free ?s - slot)
    (clear ?x - item)              ; nothing rests on ?x and it is not held
    (holding ?x - item)
    (hand-empty)
    (localized ?x - item)          ; pose known well enough to grasp
    (open ?p - place)              ; place accepts items (papers are always open)
    (lid-closed ?p - place)        ; box with a closed lid
    (fits ?x - item ?p - place))   ; geometry check done locally (height vs. shelf clearance, size vs. box)

  (:action observe                 ; new view / external camera
    :parameters (?x - item)
    :precondition (hand-empty)
    :effect (localized ?x))

  (:action pick-from-table
    :parameters (?x - item)
    :precondition (and (on-table ?x) (clear ?x) (hand-empty) (localized ?x))
    :effect (and (holding ?x) (not (on-table ?x)) (not (clear ?x)) (not (hand-empty))))

  (:action pick-from-item
    :parameters (?x - item ?y - item)
    :precondition (and (on ?x ?y) (clear ?x) (hand-empty) (localized ?x))
    :effect (and (holding ?x) (clear ?y) (not (on ?x ?y)) (not (clear ?x)) (not (hand-empty))))

  (:action pick-from-place
    :parameters (?x - item ?s - slot ?p - place)
    :precondition (and (in ?x ?p) (occupies ?x ?s) (slot-of ?s ?p) (clear ?x) (hand-empty) (localized ?x))
    :effect (and (holding ?x) (slot-free ?s) (not (in ?x ?p)) (not (occupies ?x ?s))
                 (not (clear ?x)) (not (hand-empty))))

  (:action place-on-table
    :parameters (?x - item)
    :precondition (holding ?x)
    :effect (and (on-table ?x) (clear ?x) (hand-empty) (not (holding ?x))))

  (:action place-on-item
    :parameters (?x - item ?y - item)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (on ?x ?y) (clear ?x) (hand-empty) (not (holding ?x)) (not (clear ?y))))

  (:action place-in
    :parameters (?x - item ?s - slot ?p - place)
    :precondition (and (holding ?x) (slot-of ?s ?p) (slot-free ?s) (open ?p) (fits ?x ?p))
    :effect (and (in ?x ?p) (occupies ?x ?s) (clear ?x) (hand-empty)
                 (not (holding ?x)) (not (slot-free ?s))))

  (:action open-lid                ; delete this action if the robot has no such skill
    :parameters (?p - place)
    :precondition (and (lid-closed ?p) (hand-empty))
    :effect (and (open ?p) (not (lid-closed ?p))))
)
