"""L0: граничные случаи состава и открытия отрезка (аудит 06.10.2026)."""
from tests.test_l0 import AWAY, HOME, _nodes, lineup_start


def _attack_after(faceoff):
    return lineup_start(1) + [
        faceoff,
        dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=2.0, x=5.0, y=1.0),
        dict(event_type="entry_success", actor_id="H1", actor_team_id=HOME, period=1, clock_s=2.5, x=8.0, y=1.0),
        dict(event_type="shot_on_target", actor_id="H1", actor_team_id=HOME, other_id="AG", other_team_id=AWAY,
             period=1, clock_s=3.0, x=15.0, y=1.0),
        dict(event_type="whistle", period=1, clock_s=5.0)]


def test_actorless_faceoff_opens_segment_like_attributed_faceoff():
    # Вбрасывание по определению открывает отрезок: без исполнителя и команды оно не создаёт
    # барьер цепи, а первый узел получает тот же контекст прибытия и опору времени.
    attributed = dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0",
                      other_team_id=AWAY, period=1, clock_s=0.0, x=0.0, y=0.0)
    actorless = dict(event_type="faceoff", period=1, clock_s=0.0, x=0.0, y=0.0)
    ref, _ = _nodes(_attack_after(attributed))
    nd, q = _nodes(_attack_after(actorless))
    first = nd[nd.primary == "touch"].iloc[0]
    ref_first = ref[ref.primary == "touch"].iloc[0]
    assert first.arrival == "after_faceoff"
    assert not first.chain_gap_before
    assert first.t_ref == ref_first.t_ref
    assert q["actorless_faceoff_openers"] == 1
    assert "faceoff" not in q["actorless_no_team_by_type"]


def test_manpower_above_six_is_replaced_by_reconstruction():
    # n_home/n_away включают вратаря, на стороне не больше шести: 7 — ошибка разметки.
    rows = lineup_start(1) + [
        dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
             period=1, clock_s=0.0, x=0.0, y=0.0),
        dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=2.0, x=5.0, y=1.0, n_home=7.0),
        dict(event_type="whistle", period=1, clock_s=5.0)]
    nd, q = _nodes(rows)
    touch = nd[nd.primary == "touch"].iloc[0]
    assert touch.own_sk == 5
    assert touch.mp == "EV5"
    assert q["nodes_n_out_of_range_replaced"] >= 1
