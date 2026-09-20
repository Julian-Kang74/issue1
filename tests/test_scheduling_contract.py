"""Behavioral checks for issue #52306 and the encoder-decoder regression."""

from types import SimpleNamespace

import pytest

from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.core.sched.scheduler import Scheduler


def schedule_item(
    *,
    computed: int,
    budget: int,
    item_start: int,
    item_length: int,
    encoder_decoder: bool = False,
) -> tuple[list[int], int]:
    position = PlaceholderRange(offset=item_start, length=item_length)
    request = SimpleNamespace(
        has_encoder_inputs=True,
        request_id="example",
        mm_features=[
            SimpleNamespace(
                mm_position=position,
                identifier="item-0",
                modality="text" if encoder_decoder else "image",
            )
        ],
    )
    cache = SimpleNamespace(
        check_and_update_cache=lambda _request, _index: False,
        can_allocate=lambda _request, _index, _budget, _scheduled: True,
    )
    scheduler = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_chunked_mm_input=True),
        is_encoder_decoder=encoder_decoder,
        ec_connector=None,
        encoder_cache_manager=cache,
    )
    inputs, scheduled, _, _ = Scheduler._try_schedule_encoder_inputs(
        scheduler, request, computed, budget, encoder_compute_budget=100_000
    )
    return inputs, scheduled


@pytest.mark.parametrize(
    ("computed", "budget", "expected"),
    [(0, 500, 400), (400, 500, 0), (400, 800, 800)],
)
def test_multimodal_item_is_not_split(computed: int, budget: int, expected: int):
    _, scheduled = schedule_item(
        computed=computed, budget=budget, item_start=400, item_length=800
    )
    assert scheduled == expected


def test_encoder_decoder_input_does_not_block_decoder_prefill():
    inputs, scheduled = schedule_item(
        computed=0,
        budget=100,
        item_start=0,
        item_length=200,
        encoder_decoder=True,
    )
    assert inputs == [0]
    assert scheduled == 100
