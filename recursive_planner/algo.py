# Get Goal
# Recursive bad boy:
from agent import Brain
from newton import get_action

net = Brain()


def algo(obs, goal):
    # Put obs, goal into net
    ret = net(obs, goal)

    # if obs → general goal < 0.5 # safeguard
    if ret.gen_poss < 0.5:
        # Return failure
        return "failure"

    # If can act >=90%
    if ret.this_turn_poss >= 0.9:
        # Get action
        act = get_action(net, obs, goal)
        # Do actions
        # Check action
        # Take action note
        # Return check plan


# Get_action
# Do actions
# Check action
# Take action note
# Return check plan
# Else
# Get Midpoint
# Take plan note of Call with (obs, midpoint)
# If this one fails, still try the next one…
# Take plan note and Return Call with (current state, goal)
