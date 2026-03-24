# Nanumsquare font file location
_Exported on 3/22/2026 at 17:50:28 GMT+9 from Cursor (2.6.19)_

---

**User**

@/Users/hyeon-yongjeong/.cursor/projects/Volumes-Sandisk-code-arduino-stuff-Stewart-Platform/terminals/6.txt:785-787 
there should be nanumsquare font file somewhere in /Volume/Sandisk/code.
find the font file and copy it over here to resolve error msg

---

**Cursor**

Reasonability check: the path is probably `/Volumes/Sandisk/code`, and copying a font file will only help if the warning is caused by Matplotlib not finding the intended NanumSquare font rather than the font itself lacking `U+2212`. I’ll first locate any `NanumSquare` font files and check how `presentation/Fig_Gen.py` loads fonts so I can copy the right file to the right place.

I didn’t find any `NanumSquare` font file under `/Volumes/Sandisk/code` on the first pass, so I’m checking the script’s font setup now to see whether it expects a bundled font file or just a system-installed family name.

The project tree doesn’t actually contain a `NanumSquare` font file, so I’m checking standard macOS font locations next. If I find one there, I can copy it into the project; if not, the warning likely needs a code-side fallback instead.

I’m not seeing any matching font files by filename, which suggests the font may be installed under a different filename or not present at all. I’m querying the font registry next so I can tell whether copying is possible or whether the real fix is to change the font fallback.

I found actual NanumSquare font files, including `/Volumes/Sandisk/code/bezier-trajectory/NanumSquareR.otf`, but there’s an important catch: Matplotlib already sees `NanumSquare`, so this warning is probably not caused by a missing file. I’m checking whether the script disables Unicode minus; if not, that is the more scientifically sound fix for `Glyph 8722` warnings.

The font file exists, but the assumption behind the request is false: both the installed `NanumSquareR.ttf` and the `NanumSquareR.otf` under `/Volumes/Sandisk/code` do **not** contain glyph `U+2212`, so copying them will not remove this warning. The proper fix is to make Matplotlib use ASCII `-` for tick labels by setting `axes.unicode_minus = False`; I’m checking whether `paper/Fig_Gen.py` shares the same plotting setup so I can fix this consistently if you want.

Reasonability check: copying a `NanumSquare` font file is not a sound fix for this warning.

I found valid font files:
- `/Volumes/Sandisk/code/bezier-trajectory/NanumSquareR.otf`
- `/Users/hyeon-yongjeong/Library/Fonts/NanumSquareR.ttf`

But I also verified the important part: both of those files **do not contain** glyph `U+2212` (the Unicode minus sign from the warning). Matplotlib already sees `NanumSquare`, so the issue is not “font file missing here”; it is “chosen font lacks that glyph”.

So I did **not** copy the file, because that would not resolve the warning.

The correct fix is to change `presentation/Fig_Gen.py` so Matplotlib uses ASCII `-` instead of Unicode minus, e.g. add `axes.unicode_minus: False` in the `rcParams.update(...)` block. If you want, I can patch that now and re-run the script to confirm the warning is gone.

---

**User**

fix Fig_Gen.py

---

**Cursor**

Reasonability check: this is a sensible fix, and the scientifically/logically correct change is to stop Matplotlib from emitting Unicode minus with a font that lacks that glyph. I’m updating `presentation/Fig_Gen.py` rather than copying font files, then I’ll rerun the generator to verify the warning disappears.

I’ve confirmed the plotting config; I’m making the minimal fix in the Matplotlib rcParams block so only the minus rendering behavior changes.

The code change is in place. I’m rerunning the figure generator now and checking diagnostics on the edited file to make sure the warning is actually gone and no new issues were introduced.

The rcParam fix reduced the scope but didn’t fully solve it, which means there is still an explicit Unicode minus character being rendered somewhere in `Fig4`. I’m tracing that exact label/text now and will fix it directly instead of guessing.

I found the remaining source: `Fig4` uses a literal Unicode minus in the legend label (`PX4 − cmd pitch`), so `axes.unicode_minus` alone could not affect it. I’m replacing that character with a plain ASCII hyphen and rechecking the output.

`presentation/Fig_Gen.py` is fixed.

I made two changes:
- added `"axes.unicode_minus": False` in the Matplotlib `rcParams`
- replaced the literal Unicode minus in the `Fig4` legend label with ASCII `-`

I re-ran `python3 presentation/Fig_Gen.py`, and the figures regenerated with no `Glyph 8722` warnings. No linter errors were introduced.

If you want, I can apply the same fix to `paper/Fig_Gen.py` as well so both generators stay consistent.

