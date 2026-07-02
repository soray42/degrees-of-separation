# Degrees of Separation — Roadmap (pending / 第二幕)

> 配套 [`README.md`](README.md)(已完成的平台)。这里是**待办**:扩展方向、工程思路、排序、Revelio 边界。
> 本文档是 co-author 的行动清单——**工程细节不省略**(数据列/pipeline/caveat 全保留)。
> 来源:作者最新战略判断(post-pilot)+ `paper/next_stage_agenda.tex` 平台 frontier。

**一句话战略**:pilot 已经证明"没有超出 licensing 的第二机制",别再找第二机制了。真正买到的是把
**licensing 升级成"工资压缩"这个更一般机制的一个特例**。下面全部方向围绕
**"第二幕:压缩机制——prestige 何时/为何/在哪停止付钱"**展开,**①②③④全程零 Revelio**。

---

## 0. Pilot 战利品("为什么"的地基,别丢)

- ICC(ρ|CIP-2) = 0.30/0.45:学科结构真实但 modest = **已有**的东西,不是新发现。
- 亲缘共享耦合(想法 A)**已死**:faculty-flow 图上 Moran's I = +0.011,p = 0.96,跨 7 种 W 构造稳健 null。
- CIP-2 共属图 Moran's I = +1.71、p = 0.002 **看着显著**,但 leave-pair-out 一做就塌:84% 来自
  nursing/comm-disorders 单一执照对 → **licensing 换皮,被对抗验证当场逮住的假阳性**。
- Stats↔CS 双亲缘**证伪**:Stats 是 Mathematics 卫星(9.2:1)。
- STATE×OCCP 净化:nursing raw \$5k → \$0(n = 22,518)= 护理生工资**就是**其州的 RN 职业工资,非循环纠偏。
- occ_hhi ↔ 耦合**干净 null**;CS 反例:职业锁死(occ_hhi 高)却耦合**最高 0.708**;
  occ_hhi ≈ license dummy(ρ = 0.643,p = 0.007)。
- **⇒ 概念升级**:解耦的充要条件**不是"职业锁定",是"职业内工资缺乏声望可分选的方差"**。
  压缩是机制,licensing 是最锐利的特例。

---

## 1. 三条时间轴(别只盯死掉的那条)

| 轴 | 是什么 | 数据 | 免费即刻? | Revelio 边界 |
|---|---|---|---|---|
| ① 日历/cohort 时间 | Scorecard FoS pooled 2014-15→2018-19,~4–6 cohort 点;IPEDS completions 是几十年年度供给序列 | Scorecard FoS / IPEDS | ✅ 薄,做 shock 够、做趋势不够 | — |
| ② **career/经验时间**(最肥) | coupling(t) 随职业年限走 | Scorecard 1yr+4yr(+5yr);PSEO **1/5/10 年** | ✅ **不碰 prestige 侧** | 个体轨迹不行 |
| ③ prestige 时间 | 时变 SpringRank | ORCID aff.(带年份) | ❌ 逐窗独立估撞回 edge density | — |

**②自带理论行李箱——employer learning**(Farber–Gibbons 1996 / Altonji–Pierret 2001 / Lange 2007 /
Arcidiacono–Bayer–Hizmo 2010)。三种机制三种签名,**怎么出结果都有话讲**:

- prestige 是**信号** → coupling(t) 随年限**衰减**(雇主学到真本事,母校贬值);
- 是**人力资本/网络** → 持平或**增长**(elite-tail 复利);
- 你的**压缩** field → 全程钉死 ≈0(根本没东西可学)。

**③的两条活路**(暴力硬算是当初死因,别再硬算):
- 聪明力:**state-space 平滑排名**——Whole-History Rating(Coulom 2008)、TrueSkill Through Time
  (Dangauthier et al., NIPS 2007)。时间平滑先验让稀疏期借邻期的边,density 从**致命伤**降级成
  **正则化强度的选择**。⚠️ 这是个 methods 子项目,一股 **Yifeng(minimax bounds)的味道,丢给他**。
- 摆烂力(合法版):Wapman 2022 标题带 "dynamics",结论是 2011–2020 层级**极稳定** → 引它、把
  prestige 定为**准静态固定** → gap 动态 = placement 动态,识别反而更干净。有文献背书,不是认输。
- nuance:即便切窗,可行性**按 field 分**——CS/bio(年千级 hire)或撑得起 2–3 窗,小 field 免谈。

---

## 2. 排好序的方向 + 具体工程思路

### 🅢① career-time coupling(= 轴②)— 先跑这个
**判断:免费、即刻、带理论、直接延伸压缩机制。**

- **数据**:Scorecard FoS(已在手,需确认 4yr earnings 列存在,列名 ≈ `EARN_MDN_4YR`,对 data
  dictionary 核);PSEO Earnings 表(`pseoe`,via Census PSEO Explorer/API),institution × CIP ×
  degree × {y1,y5,y10}。
- **pipeline**:
  1. 每个 horizon h:placement 向量 = (institution × field) 的中位 earnings@h。
  2. merge 核心论文已有的 per-institution SpringRank(限该 field 的院系)。
  3. 每个 field f:`coupling_f(h) = Spearman ρ(prestige_i, earnings_i@h)`,跨提供 f 的院校。
  4. 得到每个 field 的短曲线 coupling_f(h)。
  5. 检验:`coupling ~ horizon × field_type`,看压缩组 vs 整合组的**形状对比**(衰减/钉零/上扬)。
- **输出**:按 field_type 分组的 coupling(t) 曲线;headline 是三种签名的形状差异。
- **⚠️ 承重 caveat**:Scorecard **同一 release 的 1yr 和 4yr 是不同 cohort** → 直接比会把 career time
  和 calendar time 搅一锅。**PSEO 同 cohort 追 1/5/10 才干净**。设计:PSEO = 干净版,Scorecard = 覆盖版,
  互为 robustness。suppression(n<30 砍格)会薄化小 field;每 field 每 horizon 要 ≥15–20 院校才有稳定 ρ。

### 🅢② 用对变量重跑那条死掉的连续律 — 最值得动手
**判断:win-win。pilot 的 null 是喂错了变量。** 机制变量**不是 occ_hhi**(已证 ≈ license dummy),
**是"职业内工资方差"**。

- **数据**:ACS PUMS **5-year**(IPUMS)。变量:`FOD1P`(本科学位领域)、`OCCP`、`INDP`、
  `WAGP`/`PERNP`(工资)、`ST`/`PUMA`、`AGEP`(经验代理)、`SCHL`(限学士+)、`SEX`/`RAC1P`(残差化)、
  `WKHP`/`WKWN`/`ESR`(限 full-time full-year)、`PWGTP`(权重)。
- **pipeline**:
  1. 样本:学士+(`SCHL≥21`)、FTFY(周工时≥35 且周数≥50)、在业、正工资。
  2. 残差化:`log(wage) ~ 州FE + 经验多项式 + 人口特征` → 取残差(剥掉地域+composition,只留格子内离散)。
  3. 每个目的职业(`OCCP`,可选 ×`ST`)算**残差 SD** = "同一职业/地点里挺过的工资散度" =
     声望可分选方差的代理。
  4. 每个 field:从 `FOD1P × OCCP` 建 field→职业经验分布。
  5. field 级"**目的地工资离散度指数**" = 用 field 的职业分布加权的职业内残差 SD。
  6. `coupling_f ~ dispersion_index_f` 跨 field 回归。预测(压缩论):离散度↑ → 耦合↑;
     离散度≈0(nursing→RN 工资钉死)→ 解耦。
- **输出**:coupling vs 离散度指数散点,nursing 在低散度/低耦合角、CS 在高散度/高耦合角,
  回归系数+拟合 = **pilot 没找到的那条连续律**。
- **⚠️ caveat**:ACS 职业内总方差是**声望可分选方差的上界**(混着 firm 效应+未观测技能+测量误差)→
  只能当**方向性**证据。ACS **无 institution** → 只验证"机制变量"(目的地侧),不碰 prestige-vs-placement
  链本身。亮了 → 律复活、错变量诊断证实;仍 null → **"聚合数据只在两极可见、Revelio 真必要"的书面证词**。
  两条路都赢。

### 🅐③ 撬开"压缩" vs "licensing" —— UK vs US teaching
**判断:刀口正对 identification 命门,但排③(先核 LEO)。** 美国软肋:licensed/工会/公共薪级三者高度
共线(teaching、social work 全占齐),分不出谁干活。

- **数据**:UK LEO(gov.uk "Graduate outcomes (LEO)")provider × subject earnings;UK prestige 用
  nursing 复现时已建的 ORCID/faculty-hiring 网络。US:Scorecard education CIP + SpringRank。
- **pipeline**:
  1. `coupling_edu^UK = ρ(prestige, LEO earnings)` 跨 UK providers。
  2. `coupling_edu^US = ρ(prestige, Scorecard earnings)` 跨 US institutions。
  3. 两边都 licensed(执照钉死);UK = 全国 pay spine(压缩拉满),US = 学区自主(方差大)。
     预测:若压缩驱动解耦,则 `|coupling^UK| < |coupling^US|`。
  4. **US 州内加强版**:按州级教师薪表刚性(州统一表 vs 学区自主,NCTQ/州政策数据)分组,
     看 education 耦合随刚性变不变。
- **输出**:UK-vs-US 对比 + 州内刚性梯度 → 把压缩从执照里分离出来。
- **⚠️ caveat**:跨国口径(不同 prestige 网络、不同 earnings 定义/货币、CAH vs CIP 分类)→ suggestive
  not decisive。**先核 LEO 能否给 provider×subject 到所需粒度、UK prestige 网能否在该粒度建**,故排后。
  (工会压缩 Freeman 1980 理论上也是压缩源,但共线同上,所以才靠对比设计。)

### 🅐④ 给 bio 翻案 —— 第二条 placement 轴(NSF SED)
**判断:对 placement-proxy 那根柱子的正面补丁,reviewer 会喜欢。** bio 现在被标"解耦",但其 placement
市场是 **PhD/医学院管线**,year-1/4 工资看不见(博士生拿 stipend)。补它的数据**公开免费**。

- **数据**:NSF NCSES **SED baccalaureate origins**(doctorate recipients 的本科来源院校 × 博士领域计数,
  via NCSES 数据表/WebCASPAR);分母用 IPEDS completions(institution×CIP 的学士授予数)。
- **pipeline**:
  1. 每个本科院校 × field 的**博士产出率** = SED 数(分子)/ IPEDS 学士数(分母),松散对齐 cohort
     (PhD 滞后学士 ~6–8 年)。
  2. `coupling_f^PhD = ρ(prestige_i, PhD产出率_i)` 跨院校。
  3. 对比 bio 在 earnings 轴(≈解耦)vs PhD 轴的耦合。
- **输出**:双轴耦合表;若 bio 在 PhD 轴**强耦合** → "decoupled" 重分类为"**耦合到了另一个(延迟的)市场**"。
- **⚠️ caveat**:SED 本科来源表可能只到 broad field,核粒度;产出率务必归一化(别用裸计数,会被大校主导)。

### 🅑⑤ 分位数耦合(顺手探针)
**判断:零成本方向探针,不是 headline。**
- **数据**:Scorecard/PSEO 的 p25/p75 列(已在手)。
- **pipeline**:每 field 分别在 p25/p50/p75 算耦合,比 `coupling(p75)` vs `coupling(p50)`。
- **输出**:若耦合随分位上升(整合 field 尤甚)→ elite-tail 通道的方向证据。
- **⚠️ caveat**:p75 ≠ 真·p99 elite tail;院校级分位是 program 内散度,粗代理。

### 🅒⑥ PERM 绿卡数据(标死了,核实前一字别写)
- **Step 0**:下一份原始 DOL/OFLC PERM disclosure 文件,**确认教育字段里到底有没有院校名**。
  没确认前不许动工。
- 若有:免费个体级"院校 × 工资 × 雇主"。selection 巨大(sponsored 移民岗切片)→ 顶多当选择样本 robustness。

### 🗑️ 倒掉的(别刨尸体)
- **behavioral 不重开**:六族数据扫描结论仍站——免费数据喂不出 within-field 够 power 的行为结果;
  UCAS 悖论线干净死透(见 `results` 里 45/46 的 no-go)。
- **cohort 轴 COVID/travel-nursing shock**:可爱但先查 data vintage 够不够到 2021,够再说;
  且 travel pay 按证书分选不按母校,大概率还是压缩故事的加固。

---

## 3. 拼起来是什么 + Revelio 只剩什么

①②③④不是四个零件,是**一个连贯的第二幕**:
**"压缩机制——prestige 何时/为何/在哪停止付钱"** = 时间维(learning,①)+ 方差维(dispersion,②)
+ 制度维(执照/薪级/工会,③)+ placement 重定义(④)。**全程零 Revelio。**

**Revelio 被挤到只剩三样独占**(给 Yu 的信要说的):
1. 个体轨迹(谁在爬、firm-to-firm 跳跃);
2. p75 以上的真·firm elite tail;
3. 职业内方差的 **prestige-可分选分解**(ACS 只给上界,Revelio 有 institution+firm 能拆干净)。

**⇒ 给 Yu 的 Revelio 理由**:"聚合数据我把第二幕全做完了,就差这三样",比"我需要多维 placement"锋利十倍
——人家一看就知道你不是来要数据的,是来收尾的。

---

## 4. 建议顺序 + 量级

1. **①** career-time coupling — 现成 Scorecard/PSEO,几天。
2. **②** 离散度指数重跑连续律 — ACS PUMS,一两周,**最值钱**。
3. **④** bio 翻案 — NSF SED + IPEDS,一周。
4. **⑤** 顺手夹在 ① 里做。
5. **③** UK-vs-US — **先核 LEO 口径**,通过再排。
6. **⑥** 仅当亲自核实 PERM 字段带院校名。

---

## 5. 平台 frontier(`paper/next_stage_agenda.tex` 的三推力,与上面互补)

静态论文是**平台**;placement 轴当前是四样待放松的东西——**单标量、institution 粒度、US-only、median-level**
(+ 与学生 beliefs 脱节)。三推力按 access 成本排序:

- **Thrust A — 跨国外部效度**:现成公开数据即可(= 上面的 ③;UK LEO + 复现 prestige 网)。
- **Thrust B — 多维、within-field placement gap(旗舰)**:**Revelio-gated**;残差机制/tail/动态问题
  都汇入此。=上面 Revelio 三独占。
- **Thrust C — 行为层**:descriptive/quasi-exp 层现在公开数据可跑(但 42–46 已证 under-powered);
  旗舰 RCT gated on 一个做 subjective-expectations 的 co-author。

**两个杠杆点**:Revelio 订阅解锁 B + 动态问题的 placement 半边(prestige-temporal 半边另需
dynamic-SpringRank 升级,edge density 是硬约束 = 轴③);subjective-expectations 合作解锁 C。

---

## 6. 即时下一步(拿起就能干)

- [ ] **① 起步**:核 Scorecard data dictionary 确认 `EARN_MDN_4YR`(及 1yr/5yr)列,搭 `coupling_f(h)`
      骨架(复用 `src/gap.py` + 现有 SpringRank);⑤ 的 p25/p75 顺手夹进来。
- [ ] **② 主攻**:拉 ACS PUMS 5-year(IPUMS),按上面 6 步建"目的地工资离散度指数",跑
      `coupling_f ~ dispersion_index_f`。这是**最值钱**、最可能复活连续律的一枪。
- [ ] **④**:找 NSF SED baccalaureate-origins 表 + IPEDS completions,核粒度,建 bio 的 PhD-轴耦合。
- [ ] **③ 前置**:核 UK LEO provider×subject 粒度 + UK prestige 网可建性(通过才排期)。
- [ ] **⑥ 前置**:下一份 PERM disclosure,确认教育字段是否含院校名(否则一字别写)。
- [ ] **协同**:数据交接给 Yifeng(见 README §4 footprint:ship raw+interim 或按 SOURCES.md 抓);
      轴③ 的 state-space 平滑排名(WHR/TTT)是 methods 子项目,**丢给 Yifeng**。
- [ ] **co-author 决策**:`paper/COAUTHOR_NOTES.md` 六项(venue / so-what / 模型去留 / UK 是否先做 /
      outreach 对象 / CS-ranking artifact 范围)。
