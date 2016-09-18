This is the baseline algorithm as described in the course page (as pseudocode).

`segment.py` is exactly the same as the pseudocode provided by Anoop, using a priority
queue to improve efficiency.  The time complexity is O(N) if my math is correct.

`segment1.py` is the naive version with time complexity O(N^2), but easier to understand.

Both versions should get the exactly same result: 80.44 for the local test data (~90
on the leaderboard, which has a different set of test data).
