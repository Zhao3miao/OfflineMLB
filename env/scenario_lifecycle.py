import copy


def _normalized_disconnect_intervals(lifecycle):
    spawn_step = int(lifecycle.get("spawn_step", 0))
    leave_step = int(lifecycle["leave_step"])
    intervals = []
    for interval in lifecycle.get("disconnect_intervals", []):
        start = max(
            spawn_step,
            int(interval.get("start_step", interval.get("start", spawn_step))),
        )
        end = min(
            leave_step,
            int(interval.get("end_step", interval.get("end", leave_step))),
        )
        if start < end:
            intervals.append((start, end))

    intervals.sort()
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def split_disconnected_lifecycles(lifecycles):
    """Turn every visible trajectory segment into an independent MLB user.

    The original trajectory may continue while a physical user is disconnected,
    but the MLB environment must not observe that interval.  A later visible
    segment receives a fresh UE id and therefore has no identity continuity from
    the policy's perspective.
    """
    lifecycles = [copy.deepcopy(item) for item in lifecycles]
    next_ue_id = max((int(item["ue_id"]) for item in lifecycles), default=0) + 1
    visible_lifecycles = []

    for lifecycle in lifecycles:
        spawn_step = int(lifecycle.get("spawn_step", 0))
        leave_step = int(lifecycle["leave_step"])
        intervals = _normalized_disconnect_intervals(lifecycle)
        visible_ranges = []
        visible_start = spawn_step
        for disconnect_start, disconnect_end in intervals:
            if visible_start < disconnect_start:
                visible_ranges.append((visible_start, disconnect_start))
            visible_start = max(visible_start, disconnect_end)
        if visible_start < leave_step:
            visible_ranges.append((visible_start, leave_step))

        original_ue_id = int(lifecycle["ue_id"])
        for segment_index, (segment_start, segment_end) in enumerate(visible_ranges):
            segment = copy.deepcopy(lifecycle)
            if segment_index == 0:
                segment_ue_id = original_ue_id
            else:
                segment_ue_id = next_ue_id
                next_ue_id += 1

            trajectory = [
                point
                for point in segment.get("trajectory", [])
                if segment_start <= int(point["step"]) < segment_end
            ]
            if not trajectory:
                continue

            segment["ue_id"] = int(segment_ue_id)
            segment["source_trajectory_ue_id"] = original_ue_id
            segment["trajectory_segment_index"] = int(segment_index)
            segment["spawn_step"] = int(segment_start)
            segment["leave_step"] = int(segment_end)
            segment["spawn_time"] = float(segment_start)
            segment["leave_time"] = float(segment_end)
            segment["start_position"] = list(trajectory[0]["position"])
            segment["trajectory"] = trajectory
            segment["disconnect_intervals"] = []
            visible_lifecycles.append(segment)

    visible_lifecycles.sort(key=lambda item: (int(item["spawn_step"]), int(item["ue_id"])))
    return visible_lifecycles
