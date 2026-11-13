import numpy as np

from vrecon.run.view import find_events, onsets


def test_onsets_ignore_jitter_and_occlusion():
    t = 60
    c = np.zeros((t, 2)) + [100.0, 50.0]
    c += np.random.default_rng(0).normal(0, 0.4, (t, 2))  # mask-edge jitter on a still object
    c[20:40, 0] += np.linspace(0, 80, 20)  # moves 80 px between frames 20 and 40
    c[40:, 0] += 80
    area = np.full(t, 1000.0)
    c[10, 0] += 15  # a frame where something covered half of it: the centroid jumps
    area[10] = 500
    start, stop = onsets(c, area=area)
    assert abs(start - 21) <= 2 and abs(stop - 39) <= 2


def test_events_find_start_stop_and_skip_set_pieces():
    t = 50
    c = np.zeros((t, 2)) + [10.0, 10.0]
    c[15:30, 0] += np.linspace(0, 60, 15)
    c[30:, 0] += 60
    tracks = {"obs_0": {"label": 1, "centroid": c, "area": np.full(t, 500.0)},
              "obj_floor": {"label": 2, "centroid": c[::-1].copy(), "area": np.full(t, 9000.0)}}
    ev = find_events(tracks, np.ones(t))
    kinds = {(e["object"], e["kind"]) for e in ev}
    assert ("obs_0", "starts moving") in kinds and ("obs_0", "stops") in kinds
    assert not any(e["object"] == "obj_floor" for e in ev)


def test_warp_times_maps_onsets():
    from vrecon.run.service import warp_times

    t = [0.0, 1.0, 2.0, 3.0]  # frames 0, 24, 48, 72 at 24 fps
    out = warp_times(t, (24, 48), (30, 66), 24.0)
    assert [round(x * 24) for x in out] == [6, 30, 66, 90]
    assert [round(x * 24) for x in warp_times(t, (24, None), (30, None), 24.0)] == [6, 30, 54, 78]


def test_hold_keys_freezes_before_start():
    import numpy as np

    from vrecon.run.service import hold_keys
    from vrecon.schema import Pose, TransformKey

    keys = [TransformKey(time=i / 24, pose=Pose(translation=(0.0, 0.0, 0.0), rotation=(float(np.cos(0.01 * i)), 0.0, 0.0, float(np.sin(0.01 * i))))) for i in range(6)]
    out = hold_keys(keys, 24.0, 3)
    assert all(k.pose == keys[3].pose for k in out[:4]) and out[5].pose == keys[5].pose
    assert all(k.pose == keys[0].pose for k in hold_keys(keys, 24.0, None))
    rot = hold_keys(keys, 24.0, 3, rotation_only=True)
    assert rot[0].pose.rotation == keys[3].pose.rotation and rot[0].pose.translation == keys[0].pose.translation
