"""Depth-checked CURRENT-to-history reprojection; no identity decisions."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np

CYAN = (20, 215, 230)
PURPLE = (225, 65, 220)
TOLERANCE_M = 0.03


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_points(points, pose_c2w, intrinsics, depth_m, tolerance_m=TOLERANCE_M):
    """Project all input points, then z-buffer before checking measured depth."""
    points = np.asarray(points, dtype=np.float64)
    pose = np.asarray(pose_c2w, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)[:3, :3]
    depth = np.asarray(depth_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape Nx3')
    if pose.shape != (4, 4) or K.shape != (3, 3) or depth.ndim != 2:
        raise ValueError('invalid camera or depth dimensions')
    if not np.isfinite(pose).all() or not np.isfinite(K).all():
        raise ValueError('nonfinite calibration')
    if not np.allclose(pose[3], [0, 0, 0, 1]) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('invalid calibration')
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError('invalid depth tolerance')
    n = len(points)
    finite = np.isfinite(points).all(axis=1)
    indices = np.flatnonzero(finite)
    cam = np.column_stack((points[finite], np.ones(finite.sum()))) @ np.linalg.inv(pose).T
    front = cam[:, 2] > 1e-6
    behind = int((~front).sum())
    indices, cam = indices[front], cam[front, :3]
    proj = cam @ K.T
    uv_float = proj[:, :2] / proj[:, 2:3]
    h, w = depth.shape
    inside = (uv_float[:, 0] >= -.5) & (uv_float[:, 0] < w-.5)
    inside &= (uv_float[:, 1] >= -.5) & (uv_float[:, 1] < h-.5)
    outside = int((~inside).sum())
    indices, cam = indices[inside], cam[inside]
    uv = np.floor(uv_float[inside] + .5).astype(np.int64)
    linear = uv[:, 1]*w + uv[:, 0]
    order = np.lexsort((indices, cam[:, 2], linear))
    ordered = linear[order]
    keep = order[np.r_[True, ordered[1:] != ordered[:-1]]] if len(order) else order
    duplicates = len(indices)-len(keep)
    indices, cam, uv = indices[keep], cam[keep], uv[keep]
    measured = depth[uv[:, 1], uv[:, 0]]
    valid = np.isfinite(measured) & (measured > 0)
    delta = cam[:, 2]-measured
    reliable = valid & (np.abs(delta) <= tolerance_m)
    occluded = valid & (delta > tolerance_m)
    conflict = valid & (delta < -tolerance_m)
    # status: 0 invalid depth; 1 reliable; 2 occluded; 3 in front of measured surface.
    status = np.zeros(len(uv), np.uint8)
    status[reliable], status[occluded], status[conflict] = 1, 2, 3
    counts = dict(input_points=n, nonfinite=int((~finite).sum()), behind_camera=behind,
                  outside_image=outside, raster_collisions=int(duplicates),
                  invalid_depth=int((~valid).sum()), reliable=int(reliable.sum()),
                  occluded=int(occluded.sum()), depth_conflict=int(conflict.sum()))
    assert sum(v for k, v in counts.items() if k != 'input_points') == n
    return dict(uv=uv, input_indices=indices, projected_depth=cam[:, 2],
                historical_depth=measured, depth_delta=delta, status=status,
                reliable_uv=uv[reliable], counts=counts)


def candidate_statistics(projected, mask):
    """Mask influences counts and framing, never projection/visibility positions."""
    mask = np.asarray(mask, dtype=bool)
    uv = projected['reliable_uv']
    support = int(mask[uv[:, 1], uv[:, 0]].sum()) if len(uv) else 0
    counts = dict(projected['counts'])
    n, reliable = counts['input_points'], counts['reliable']
    counts.update(in_mask=support, outside_mask=reliable-support,
                  comparable_fraction=reliable/n if n else None,
                  in_mask_fraction=support/reliable if reliable else None,
                  depth_conflict_fraction=counts['depth_conflict']/n if n else None)
    return counts


def union_crop(mask, uv):
    ys, xs = np.where(mask)
    if len(uv):
        xs, ys = np.r_[xs, uv[:, 0]], np.r_[ys, uv[:, 1]]
    h, w = mask.shape
    if not len(xs):
        return (0, 0, w, h)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max())+1, int(ys.min()), int(ys.max())+1
    px, py = max(18, round((x1-x0)*.12)), max(18, round((y1-y0)*.12))
    return max(0, x0-px), max(0, y0-py), min(w, x1+px), min(h, y1+py)


def text(im, value, x, y, scale=.54, color=(232, 238, 246)):
    cv2.putText(im, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def panels(rgb, mask, uv, box, width=612, height=338, candidate_hatch=False):
    x0, y0, x1, y1 = box
    raw = rgb[y0:y1, x0:x1]
    scale = min(width/raw.shape[1], height/raw.shape[0])
    rw, rh = max(1, round(raw.shape[1]*scale)), max(1, round(raw.shape[0]*scale))
    raw_resized = cv2.resize(raw, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    offx, offy = (width-rw)//2, (height-rh)//2
    left = np.full((height, width, 3), (17, 22, 30), np.uint8)
    left[offy:offy+rh, offx:offx+rw] = raw_resized
    right = left.copy()
    local_mask = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8), (rw, rh),
                            interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(local_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    viewport = right[offy:offy+rh, offx:offx+rw]
    if candidate_hatch:
        strokes = np.zeros((rh, rw), np.uint8)
        for intercept in range(-rh, rw, 14):
            cv2.line(strokes, (intercept, 0), (intercept+rh-1, rh-1),
                     255, 1, cv2.LINE_AA)
        # Clip to actual mask pixels, including holes; never hatch the contour hull.
        hatch_alpha = (strokes.astype(np.float32)/255.0 *
                       local_mask.astype(bool) * .55)[..., None]
        viewport[:] = np.rint(viewport*(1-hatch_alpha)+
                              np.asarray(CYAN)*hatch_alpha).astype(np.uint8)
    current_cover = np.zeros((rh, rw), np.uint8)
    if len(uv):
        xy = np.column_stack(((uv[:, 0]-x0+.5)*rw/(x1-x0)-.5,
                              (uv[:, 1]-y0+.5)*rh/(y1-y0)-.5))
        xy = np.rint(xy).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, rw-1)
        xy[:, 1] = np.clip(xy[:, 1], 0, rh-1)
        # Render retained points only. No convex hull, mask fill or hole completion.
        for x, y in np.unique(xy, axis=0):
            cv2.circle(current_cover, (int(x), int(y)), 2, 255, -1, cv2.LINE_AA)
        # Small display footprints make sparse samples visible; never fill a hull.
        alpha = (current_cover.astype(np.float32)/255.0*.48)[..., None]
        viewport[:] = np.rint(viewport*(1-alpha)+np.asarray(PURPLE)*alpha).astype(np.uint8)
    # Draw the candidate boundary last so it stays visible above CURRENT coverage.
    cv2.drawContours(viewport, contours, -1, CYAN, 2, cv2.LINE_AA)
    return left, right


def load_history(history):
    if not Path(history['rgb_path']).is_file() or not Path(history['depth_path']).is_file():
        raise ValueError('historical camera/depth unavailable')
    rgb_bgr = cv2.imread(str(history['rgb_path']))
    raw_depth = cv2.imread(str(history['depth_path']), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or raw_depth is None:
        raise ValueError('unreadable historical RGB or depth')
    hw = tuple(int(x) for x in history['image_hw'])
    mask = np.asarray(history['mask'], dtype=bool)
    if mask.shape != hw or raw_depth.ndim != 2:
        raise ValueError('history mask must match calibrated image grid')
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != hw:
        rgb = cv2.resize(rgb, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(raw_depth.astype(float), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    depth /= float(history['depth_scale'])
    return rgb, mask, depth


def build_projection_card(alias, points, histories, directory, h_frame,
                          snapshot_uid=None, tolerance_m=TOLERANCE_M, candidate_hatch=False):
    started = time.perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    card = np.full((1536, 1280, 3), (25, 32, 44), np.uint8)
    text(card, f'CANDIDATE {alias} | CURRENT REPROJECTED INTO HISTORY', 22, 33, .76)
    text(card, 'CYAN = candidate outline + hatch' if candidate_hatch else 'CYAN = candidate mask outline', 22, 65, .56, CYAN)
    text(card, 'PURPLE FILL = CURRENT reprojection', 610, 65, .56, PURPLE)
    text(card, 'Same CURRENT in H1 / H2 / H3. Left and right use the identical crop.', 22, 92, .53)
    records = []
    if len(histories) > 3:
        raise ValueError('expected existing H1/H2/H3 only')
    for i in range(3):
        y = 110+i*460
        cv2.line(card, (16, y), (1264, y), (74, 87, 104), 1)
        text(card, f'H{i+1} | ORIGINAL RGB', 20, y+30, .6)
        text(card, f'H{i+1} | CANDIDATE {alias} + CURRENT', 660, y+30, .6)
        if i >= len(histories):
            text(card, 'HISTORY UNAVAILABLE - no additional past view', 70, y+215, .68)
            records.append(dict(display_role=f'H{i+1}', available=False, reason='no_existing_history'))
            continue
        history = histories[i]
        if int(history['frame_idx']) > int(h_frame):
            raise ValueError('future history is forbidden')
        record = {k:v for k, v in history.items() if k != 'mask'}
        record.update(display_role=f'H{i+1}', h_frame=int(h_frame),
                      source_h_snapshot_uid=snapshot_uid)
        try:
            rgb, mask, depth = load_history(history)
            projected = project_points(points, history['pose_c2w'], history['intrinsics'], depth, tolerance_m)
            stats = candidate_statistics(projected, mask)
            box = union_crop(mask, projected['reliable_uv'])
            left, right = panels(rgb, mask, projected['reliable_uv'], box, candidate_hatch=candidate_hatch)
            card[y+43:y+381, 16:628] = left
            card[y+43:y+381, 652:1264] = right
            support = f"{stats['in_mask']}/{stats['reliable']}" if stats['reliable'] else 'N/A'
            text(card, f"Comparable {stats['reliable']}/{stats['input_points']} | In candidate mask {support}"
                 + f" | Front-depth conflict {stats['depth_conflict']}/{stats['input_points']}",
                 22, y+407, .52)
            if stats['reliable']:
                status_text = ('CURRENT: purple coverage. CANDIDATE: cyan hatch inside mask; holes remain empty.' if candidate_hatch else
                               'CURRENT: translucent sample coverage. CANDIDATE: outline only. Gaps are not filled.')
            else:
                status_text = 'NO COMPARABLE PROJECTION - absence of purple coverage does NOT imply NEW.'
            text(card, status_text, 22, y+438, .51, (248, 207, 126) if not stats['reliable'] else (177, 194, 216))
            rawfile = directory/f'projection_{alias}_H{i+1}.npz'
            np.savez_compressed(rawfile, **{k:v for k,v in projected.items() if k != 'counts'})
            record.update(available=True, comparable=bool(stats['reliable']), counts=stats,
                          crop_xyxy=list(box), tolerance_m=tolerance_m,
                          mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                          rgb_sha256=sha(history['rgb_path']), depth_sha256=sha(history['depth_path']),
                          projection_path=rawfile.name, projection_sha256=sha(rawfile))
        except (ValueError, OSError) as exc:
            text(card, 'PROJECTION UNAVAILABLE - use the original evidence', 65, y+215, .64)
            record.update(available=False, reason=str(exc))
        records.append(record)
    text(card, 'Hidden points may be occluded, out of view or depth-inconsistent. Partial coverage is allowed.',
         22, 1516, .53)
    target = directory/f'projection_{alias}.jpg'
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(card, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError('JPEG encoding failed')
    target.write_bytes(buf.tobytes())
    return dict(alias=alias, role='CURRENT REPROJECTION '+alias, path=target.name,
                sha256=sha(target), width=1280, height=1536,
                current_input_points=len(points), source_h_snapshot_uid=snapshot_uid,
                renderer='current_translucent_coverage_candidate_outline_v2',
                overlay_alpha=.48, display_sample_radius_px=2,
                histories=records, seconds=time.perf_counter()-started)



def build_integrated_candidate_card(alias, original_path, points, histories, directory,
                                    h_frame, snapshot_uid=None, tolerance_m=TOLERANCE_M, candidate_hatch=False):
    """Replace only the candidate RGB row; keep target-only and shared 3D panels."""
    started = time.perf_counter()
    directory = Path(directory)
    original_path = Path(original_path)
    original_sha = sha(original_path)
    # The large card and raw projection arrays remain available for inspection,
    # but are not extra VLM attachments.
    meta = build_projection_card(alias, points, histories, directory, h_frame,
                                 snapshot_uid, tolerance_m, candidate_hatch=candidate_hatch)
    diagnostic = dict(path=meta['path'], sha256=meta['sha256'])
    raw = cv2.imread(str(original_path))
    if raw is None or raw.shape[:2] != (1024, 1024):
        raise ValueError('expected the frozen 1024x1024 candidate card')
    card = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    original_lower = card[284:].copy()
    card[:78] = (28, 31, 36)
    text(card, f'CANDIDATE {alias} | SAME H1/H2/H3; CURRENT REPROJECTION', 14, 25, .64)
    text(card, 'PURPLE FILL = CURRENT', 14, 48, .53, PURPLE)
    text(card, 'CYAN HATCH = CANDIDATE' if candidate_hatch else 'CYAN OUTLINE = CANDIDATE', 414, 48, .53, CYAN)
    for i, record in enumerate(meta['histories']):
        x = 12+i*336
        if i >= len(histories):
            text(card, f'H{i+1} UNAVAILABLE', x+3, 69, .44)
            continue
        history = histories[i]
        if record['available']:
            rgb, mask, _ = load_history(history)
            with np.load(directory/record['projection_path'], allow_pickle=False) as data:
                uv = data['reliable_uv']
            box = record['crop_xyxy']
            title = f'H{i+1} RGB + PROJ | {len(uv)} reliable' if len(uv) else f'H{i+1} | NO RELIABLE PROJ'
        else:
            # Depth can be missing while the original RGB/mask remains usable.
            bgr = cv2.imread(str(history['rgb_path']))
            if bgr is None:
                raise ValueError('unreadable original history RGB')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = np.asarray(history['mask'], dtype=bool)
            if rgb.shape[:2] != mask.shape:
                rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            uv = np.empty((0, 2), dtype=int)
            box = union_crop(mask, uv)
            title = f'H{i+1} | PROJ UNAVAILABLE'
        _, overlay = panels(rgb, mask, uv, box, width=328, height=206, candidate_hatch=candidate_hatch)
        card[78:284, x:x+328] = overlay
        text(card, title, x+3, 69, .40)
    assert np.array_equal(card[284:], original_lower), 'target-only/3D layout changed'
    target = directory/f'candidate_{alias}_projection.jpg'
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(card, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError('JPEG encoding failed')
    target.write_bytes(buf.tobytes())
    meta.update(role='CANDIDATE '+alias, path=target.name, sha256=sha(target),
                width=1024, height=1024,
                renderer='integrated_candidate_rgb_projection_hatch_v4' if candidate_hatch else 'integrated_candidate_rgb_projection_v3',
                candidate_hatch=bool(candidate_hatch), hatch_spacing_px=14 if candidate_hatch else None,
                hatch_alpha=.55 if candidate_hatch else None, hatch_line_width_px=1 if candidate_hatch else None,
                original_candidate_path=original_path.name,
                original_candidate_sha256=original_sha, diagnostic_card=diagnostic,
                preserved_decoded_rows=[284, 1024], seconds=time.perf_counter()-started)
    return meta
