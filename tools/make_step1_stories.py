#!/usr/bin/env python3
"""Write a deterministic synthetic Chinese children's-story corpus.

The text is original template prose for the Step 1 sandbox. It is not copied
from any published book. Eighty short books, simple vocabulary, paragraphs of
continuous narrative rather than chat fragments.

Public-domain downloads were not required for this generator. Re-run:

    python3 tools/make_step1_stories.py --out data/step1/corpus.txt --bytes 60000000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

HEROES = [
    "小猫", "小狗", "小兔", "小熊", "小鸟", "小鸭", "小羊", "小鱼",
    "小鹿", "小狐狸", "小松鼠", "小青蛙", "小乌龟", "小猪", "小象", "小猴",
]
FRIENDS = ["奶奶", "爷爷", "小朋友", "老师", "邻居", "风", "月亮", "太阳"]
PLACES = [
    "厨房", "院子", "河边", "森林", "学校", "花园", "桥边", "山坡",
    "菜地", "图书馆", "操场", "窗台", "门口", "草地上", "井边", "果园",
]
OBJECTS = [
    "一碗鱼汤", "一个红苹果", "一把雨伞", "一只风筝", "一个皮球", "一封信",
    "一篮草莓", "一壶热水", "一张图画", "一双小鞋", "一盏灯", "一本旧书",
    "一条围巾", "一块暖石头", "一串葡萄", "一只纸船",
]
ACTIONS = [
    "轻轻走过去", "停下来看了很久", "把东西抱在怀里", "问旁边的朋友",
    "先想了想", "笑出了声", "小心翼翼地伸手", "绕着圈跑了两步",
]
RESULTS = [
    "心里暖和起来", "决定明天再来", "把看见的事告诉了朋友", "没有弄坏任何东西",
    "学会等一等", "把地方收拾干净", "和朋友一起回家", "在本子上画了下来",
]
TIMES = ["一天早上", "傍晚", "下雨的下午", "放学以后", "月亮出来的时候", "春天的中午", "冬天的早晨", "暑假的第一天"]
QUOTES = [
    "我们慢慢来。", "这个可以分给你。", "我有一点害怕，可是我想试试。",
    "先洗手再吃。", "明天我们还来这里。", "谢谢你等我。", "看，天晴了。",
    "不要着急，汤还很烫。", "我把故事讲给你听。", "小碗给小的，大碗给大的。",
]


def paragraph(rng: random.Random, hero: str, motif: bool) -> str:
    friend = rng.choice(FRIENDS)
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    when = rng.choice(TIMES)
    if motif and hero == "小猫" and rng.random() < 0.35:
        obj = "一碗鱼汤"
        place = rng.choice(["厨房", "桌子上", "院子"])
        opening = f"{when}，小猫看见桌子上有一碗鱼汤，它闻了闻，说好香。"
    else:
        opening = f"{when}，{hero}来到{place}，看见{obj}。"
    lines = [
        opening,
        f"{hero}{rng.choice(ACTIONS)}。{friend}也在旁边，安静地看着。",
        f"{hero}说：「{rng.choice(QUOTES)}」",
        f"于是{hero}和{friend}一起{rng.choice(ACTIONS)}，{rng.choice(RESULTS)}。",
        f"风从{place}吹过来，叶子轻轻响。{hero}记得要把{obj}放回原来的地方。",
    ]
    if rng.random() < 0.5:
        lines.append(f"回到家的路上，{hero}还在想这件事。它觉得今天没有白来。")
    return "".join(lines)


def write_corpus(path: Path, target_bytes: int, seed: int, books: int) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for i in range(books):
            hero = HEROES[i % len(HEROES)]
            title = f"第{i + 1}本　{hero}的一天\n\n"
            f.write(title)
            written += len(title.encode("utf-8"))
            chapter = 1
            while written < target_bytes * (i + 1) // books:
                head = f"第{chapter}章\n"
                f.write(head)
                written += len(head.encode("utf-8"))
                for _ in range(rng.randint(6, 10)):
                    text = paragraph(rng, hero, motif=True) + "\n"
                    f.write(text)
                    written += len(text.encode("utf-8"))
                f.write("\n")
                written += 1
                chapter += 1
                if chapter > 400:
                    break
    print(f"wrote {path} ({written} bytes, {books} books)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/step1/corpus.txt"))
    p.add_argument("--bytes", type=int, default=60_000_000)
    p.add_argument("--books", type=int, default=80)
    p.add_argument("--seed", type=int, default=21)
    args = p.parse_args()
    write_corpus(args.out, args.bytes, args.seed, args.books)


if __name__ == "__main__":
    main()
