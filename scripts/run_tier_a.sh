#!/bin/sh
# Возобновляемая цепочка уровня A (см. README, «Как воспроизвести»): каждый шаг без --force берёт готовое из
# кэша (ключ: данные, конфигурация, код, входы), поэтому после обрыва или выключения машины
# досчитывается только недостающее. Повторный вызов того же шага проверяет загрузку из кэша.
# Останов на первом сбое; статусы шагов — logs/tier_a.status, журналы — logs/tier_a.<шаг>.log.
#
#   sh scripts/run_tier_a.sh            # до L5 включительно + отладочный L6 (fast)
#   sh scripts/run_tier_a.sh full       # то же + сдаваемый L6 (бутстрэп B = 200, часы)
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
PY=.venv/bin/python
LOG=logs/tier_a
STATUS=$LOG.status
mkdir -p logs

step() {
  name=$1; shift
  start=$(date +%s)
  echo "$(date '+%d.%m %H:%M:%S') START $name" >> "$STATUS"
  "$@" > "$LOG.$name.log" 2>&1
  code=$?
  echo "$(date '+%d.%m %H:%M:%S') END $name exit=$code seconds=$(( $(date +%s) - start ))" >> "$STATUS"
  if [ $code -ne 0 ]; then
    echo "CHAIN STOPPED at $name (exit 2 = провален гейт качества)" >> "$STATUS"
    exit $code
  fi
}

# Повтор шага обязан взять результат из кэша: промах означает дефект ключа, а не повод молча
# пересчитывать слой часами (так в 06.10 нашёлся вечный промах кэша L1).
expect_cached() {
  name=$1; pattern=$2
  if ! grep -q "$pattern" "$LOG.$name.log"; then
    echo "$(date '+%d.%m %H:%M:%S') CHAIN STOPPED at $name: повтор не взят из кэша" >> "$STATUS"
    exit 5
  fi
}

step prepare        $PY -m datamatch.cli prepare
step xg             $PY -m datamatch.cli xg
step xg_cached      $PY -m datamatch.cli xg
expect_cached xg_cached "выбор xG актуален, загружен из"
step value          $PY -m datamatch.cli value
step value_cached   $PY -m datamatch.cli value
expect_cached value_cached "current selection loaded from cache"
step verify_l2      $PY scripts/verify_l2_artifacts.py
step outer_audit    $PY scripts/audit_frozen_layers.py
step contrib        $PY -m datamatch.cli contrib
step shrink         $PY -m datamatch.cli shrink
step ratings_fast   $PY -m datamatch.cli ratings --mode fast
step display_fast   $PY scripts/apply_display_rules.py --mode fast
step display_fast_ok $PY scripts/apply_display_rules.py --mode fast --check
if [ "$1" = "full" ]; then
  step ratings_full $PY -m datamatch.cli ratings --mode full
  step display_full $PY scripts/apply_display_rules.py --mode full
  step display_full_ok $PY scripts/apply_display_rules.py --mode full --check
fi
# Карточка игрока (METHODOLOGY 11): отдельные поля вне рейтинга; читают готовые кэши и test_ratings_detail.csv.
step card_war         $PY scripts/card_war.py
step card_roles       $PY scripts/card_roles.py
step card_situational $PY scripts/card_situational.py
step card_penalties   $PY scripts/card_penalties.py
step card_phys        $PY scripts/card_phys.py
step card_faceoffs    $PY scripts/card_faceoffs.py
step card_club        $PY scripts/card_club.py
step card_report      $PY scripts/card_report.py
echo "$(date '+%d.%m %H:%M:%S') CHAIN DONE" >> "$STATUS"
