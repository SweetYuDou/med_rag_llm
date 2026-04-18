import React, { useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Empty,
  Form,
  Input,
  Select,
  Spin,
  message,
} from 'antd';
import {
  ArrowRightOutlined,
  ExperimentOutlined,
  MedicineBoxOutlined,
  NodeIndexOutlined,
  RobotOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons';
import axios from 'axios';

const { TextArea } = Input;

const promptSuggestions = [
  '发热两天，伴有咳嗽、流涕和喉咙痛',
  '肚子疼，还恶心，昨晚开始更明显',
  '胸口发闷，走快一点就喘',
  '头痛头晕一周，最近特别没精神',
];

const durationOptions = [
  { value: 'within_24h', label: '24 小时内' },
  { value: '1_3_days', label: '1 到 3 天' },
  { value: 'over_1_week', label: '超过 1 周' },
  { value: 'over_1_month', label: '超过 1 个月' },
];

const severityOptions = [
  { value: 'mild', label: '轻度' },
  { value: 'moderate', label: '中度' },
  { value: 'severe', label: '重度' },
];

const triageToneMap = {
  建议立即就医: 'danger',
  建议尽快门诊: 'warning',
  建议门诊随访: 'calm',
};

const probabilityTone = (value) => {
  const number = parseInt(value, 10);
  if (Number.isNaN(number)) {
    return 'neutral';
  }
  if (number >= 80) {
    return 'high';
  }
  if (number >= 60) {
    return 'medium';
  }
  return 'low';
};

const buildLeadSummary = (result) => {
  if (!result) {
    return '';
  }

  const disease = result.top_diseases?.[0]?.name || '当前症状组合';
  const department = result.recommended_department?.primary || '全科医学科';
  const symptomCount = result.recognized_symptoms?.length || 0;
  const testCount = result.recommended_tests?.length || 0;

  return `系统当前基于 ${symptomCount} 个已命中症状节点，优先建议沿着 ${disease} 相关方向继续排查，首诊建议为 ${department}。同时整理出 ${testCount} 项可优先考虑的检查与后续处置线索。`;
};

const buildUserQuery = (values) => {
  const symptoms = values.symptoms?.trim() || '';
  const history = values.history?.trim() || '';

  if (!history) {
    return symptoms;
  }

  return `主要症状：${symptoms}\n既往史/当前用药：${history}`;
};

const ChipSet = ({ items, tone = 'neutral', emptyText = '暂无相关信息' }) => {
  if (!items?.length) {
    return <span className="empty-copy">{emptyText}</span>;
  }

  return (
    <div className="chip-set">
      {items.map((item) => (
        <span key={item} className={`chip chip-${tone}`}>
          {item}
        </span>
      ))}
    </div>
  );
};

const StatTile = ({ label, value, caption, tone = 'ink' }) => (
  <div className={`stat-tile stat-tile-${tone}`}>
    <span className="stat-label">{label}</span>
    <strong className="stat-value">{value}</strong>
    {caption ? <span className="stat-caption">{caption}</span> : null}
  </div>
);

const Section = ({ eyebrow, title, children }) => (
  <section className="report-section">
    <div className="section-head">
      <span className="section-eyebrow">{eyebrow}</span>
      <h3 className="section-title">{title}</h3>
    </div>
    {children}
  </section>
);

const ReportList = ({ items, emptyText = '暂无内容' }) => {
  if (!items?.length) {
    return <span className="empty-copy">{emptyText}</span>;
  }

  return (
    <ul className="report-list">
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
};

const NodeRibbon = ({ nodes }) => {
  if (!nodes?.length) {
    return <span className="empty-copy">暂无命中的图谱节点</span>;
  }

  return (
    <div className="node-ribbon">
      {nodes.map((node, index) => (
        <div key={`${node.mention}-${node.node_name}-${index}`} className="node-card">
          <div className="node-path">
            <span className="node-mention">{node.mention}</span>
            <ArrowRightOutlined />
            <span className="node-name">{node.node_name}</span>
          </div>
          <div className="node-meta">
            <span>{node.node_type}</span>
            <span>{node.polarity === 'negative' ? '否定' : '肯定'}</span>
          </div>
        </div>
      ))}
    </div>
  );
};

const CandidateCard = ({ item, index }) => (
  <article className="candidate-card">
    <div className="candidate-rank">
      <span>{String(index + 1).padStart(2, '0')}</span>
      <span className={`probability-pill probability-${probabilityTone(item.probability)}`}>
        {item.probability}
      </span>
    </div>

    <div className="candidate-main">
      <div className="candidate-title-row">
        <div>
          <h4 className="candidate-title">{item.name}</h4>
          <p className="candidate-reason">{item.reason}</p>
        </div>
      </div>

      <div className="candidate-grid">
        <div className="info-panel">
          <span className="panel-label">匹配到的症状</span>
          <ChipSet items={item.matched_symptoms} tone="teal" emptyText="暂无稳定匹配项" />
        </div>
        <div className="info-panel">
          <span className="panel-label">建议继续追问</span>
          <ChipSet
            items={item.possible_additional_symptoms}
            tone="gold"
            emptyText="暂无补充追问项"
          />
        </div>
        <div className="info-panel">
          <span className="panel-label">建议科室</span>
          <ChipSet items={item.recommended_department} tone="neutral" emptyText="暂无明确科室" />
        </div>
        <div className="info-panel">
          <span className="panel-label">可优先检查</span>
          <ChipSet items={item.recommended_checks} tone="rust" emptyText="暂无明确检查" />
        </div>
        <div className="info-panel candidate-wide">
          <span className="panel-label">治疗线索</span>
          <ChipSet items={item.treatment_options} tone="slate" emptyText="暂无明确治疗线索" />
        </div>
      </div>
    </div>
  </article>
);

const App = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [form] = Form.useForm();

  const triageTone = triageToneMap[result?.triage_level] || 'calm';
  const leadSummary = useMemo(() => buildLeadSummary(result), [result]);

  const handleSuggestion = (suggestion) => {
    form.setFieldsValue({ symptoms: suggestion });
  };

  const resetForm = () => {
    form.resetFields();
    setResult(null);
  };

  const onFinish = async (values) => {
    setLoading(true);
    setResult(null);

    try {
      const payload = {
        ...values,
        query: buildUserQuery(values),
      };
      const response = await axios.post('/api/diagnose', payload, { timeout: 30000 });
      setResult(response.data);
      message.success('综合报告已生成');
    } catch (error) {
      const apiMessage = error?.response?.data?.error;
      message.error(apiMessage || '请求失败，请确认后端服务已经启动。');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />

      <header className="app-header">
        <div className="header-copy">
          <div className="header-badges">
            <span className="header-badge">Clinical Graph Workspace</span>
            <span className="header-badge muted">单页综合报告</span>
          </div>
          <h1 className="display-title">常见病辅助诊断工作台</h1>
          <p className="header-text">
            用户输入症状后，系统会串联“节点识别、知识图谱检索、约束生成”三段链路，最后输出一份更接近临床阅览习惯的整合报告。
          </p>
        </div>

        <div className="header-aside">
          <div className="header-note">
            <MedicineBoxOutlined />
            <div>
              <strong>图谱范围</strong>
              <span>常见病、常见症状、检查与科室</span>
            </div>
          </div>
          <div className="header-note">
            <NodeIndexOutlined />
            <div>
              <strong>推理路径</strong>
              <span>4B 节点抽取 → KG 检索 → 8B 约束生成</span>
            </div>
          </div>
        </div>
      </header>

      <main className="workspace">
        <section className="panel intake-panel">
          <div className="panel-kicker">输入面板</div>
          <h2 className="panel-title">先整理症状，再生成报告</h2>
          <p className="panel-copy">
            这里保留了最少但够用的输入项。提交时会自动把“主要症状”和“既往史/当前用药”拼成一条完整用户 query，再送到后端做节点识别和图谱检索。
          </p>

          <div className="quick-notes">
            <div className="quick-note">
              <ExperimentOutlined />
              <div>
                <strong>更适合口语输入</strong>
                <span>支持“发烧、咳嗽、浑身疼”这类表达</span>
              </div>
            </div>
            <div className="quick-note">
              <SafetyCertificateOutlined />
              <div>
                <strong>风险优先</strong>
                <span>如果有危险信号，会优先提示尽快就医</span>
              </div>
            </div>
          </div>

          <div className="suggestion-board">
            {promptSuggestions.map((item) => (
              <button
                key={item}
                type="button"
                className="suggestion-chip"
                onClick={() => handleSuggestion(item)}
              >
                {item}
              </button>
            ))}
          </div>

          <Form
            form={form}
            layout="vertical"
            onFinish={onFinish}
            initialValues={{
              duration: '1_3_days',
              severity: 'moderate',
            }}
          >
            <Form.Item
              name="symptoms"
              label="主要症状"
              rules={[{ required: true, message: '请先输入主要症状。' }]}
            >
              <TextArea
                autoSize={{ minRows: 6, maxRows: 9 }}
                className="composer-input"
                placeholder="例如：发热两天，咳嗽比较频繁，喉咙痛，今天开始有点喘。"
              />
            </Form.Item>

            <Form.Item name="history" label="既往史 / 当前用药">
              <TextArea
                autoSize={{ minRows: 4, maxRows: 6 }}
                className="composer-input"
                placeholder="例如：有鼻炎史，无高血压，今天自行服用过退烧药。"
              />
            </Form.Item>

            <div className="form-split">
              <Form.Item name="duration" label="病程">
                <Select options={durationOptions} popupMatchSelectWidth={false} />
              </Form.Item>
              <Form.Item name="severity" label="严重程度">
                <Select options={severityOptions} popupMatchSelectWidth={false} />
              </Form.Item>
            </div>

            <div className="action-footer">
              <p className="action-copy">
                如存在呼吸困难、胸痛、意识异常等危险信号，系统会直接提高分诊优先级。
              </p>
              <div className="action-group">
                <Button onClick={resetForm}>清空</Button>
                <Button
                  type="primary"
                  htmlType="submit"
                  icon={<RobotOutlined />}
                  loading={loading}
                >
                  生成综合报告
                </Button>
              </div>
            </div>
          </Form>
        </section>

        <section className="panel report-panel">
          {!loading && !result && (
            <div className="report-placeholder">
              <div className="placeholder-mark">
                <MedicineBoxOutlined />
              </div>
              <h3>等待输入后生成报告</h3>
              <p>
                提交症状后，这里会按“综合结论、候选疾病、图谱证据、后续建议”的顺序生成一整页结果。
              </p>
            </div>
          )}

          {loading && (
            <div className="report-placeholder loading-state">
              <Spin size="large" />
              <h3>正在整理综合报告</h3>
              <p>正在进行节点识别、知识图谱检索、候选疾病排序与建议生成。</p>
            </div>
          )}

          {!loading && result && (
            <div className="report-stack">
              <section className="report-cover">
                <div className="cover-copy">
                  <span className="cover-kicker">
                    <RobotOutlined />
                    图谱约束综合结论
                  </span>
                  <span className={`triage-pill triage-pill-${triageTone}`}>
                    {result.triage_level || '门诊评估'}
                  </span>
                  <h2 className="cover-title">
                    {result.top_diseases?.[0]?.name || '需要进一步临床评估'}
                  </h2>
                  <p className="cover-summary">{result.advice}</p>
                  <p className="cover-lead">{leadSummary}</p>
                </div>

                <div className="cover-stats">
                  <StatTile
                    label="首诊建议"
                    value={result.recommended_department?.primary || '全科医学科'}
                    caption="系统当前优先推荐科室"
                    tone="teal"
                  />
                  <StatTile
                    label="候选疾病"
                    value={String(result.top_diseases?.length || 0)}
                    caption="当前报告保留的重点方向"
                    tone="gold"
                  />
                  <StatTile
                    label="推荐检查"
                    value={String(result.recommended_tests?.length || 0)}
                    caption="已整理出的检查线索"
                    tone="rust"
                  />
                </div>
              </section>

              <Section eyebrow="综合结论" title="先去哪看、先做什么">
                <div className="report-grid report-grid-two">
                  <div className="info-panel">
                    <span className="panel-label">首诊科室</span>
                    <p className="focus-line">
                      {result.recommended_department?.primary || '全科医学科'}
                    </p>
                    <p className="supporting-copy">
                      {result.recommended_department?.reason || '当前优先级由候选疾病和图谱命中结果共同决定。'}
                    </p>
                    <ChipSet
                      items={result.recommended_department?.alternatives}
                      tone="neutral"
                      emptyText="暂无备选科室"
                    />
                  </div>

                  <div className="info-panel">
                    <span className="panel-label">系统摘要</span>
                    <p className="focus-line">{result.warning || '当前未出现额外高风险提示。'}</p>
                    <p className="supporting-copy">{result.disclaimer}</p>
                  </div>
                </div>
              </Section>

              <Section eyebrow="候选疾病" title="当前更值得优先排查的方向">
                {result.top_diseases?.length ? (
                  <div className="candidate-stack">
                    {result.top_diseases.map((item, index) => (
                      <CandidateCard key={`${item.name}-${index}`} item={item} index={index} />
                    ))}
                  </div>
                ) : (
                  <div className="section-empty">
                    <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="当前没有足够证据召回候选疾病。" />
                  </div>
                )}
              </Section>

              <Section eyebrow="图谱证据" title="系统在图谱里命中了哪些节点">
                <div className="report-grid report-grid-two">
                  <div className="info-panel">
                    <span className="panel-label">识别出的节点</span>
                    <NodeRibbon nodes={result.recognized_nodes} />
                  </div>

                  <div className="info-panel">
                    <span className="panel-label">已命中的症状节点</span>
                    <ChipSet
                      items={result.recognized_symptoms}
                      tone="teal"
                      emptyText="暂无稳定命中的症状节点"
                    />
                  </div>

                  <div className="info-panel">
                    <span className="panel-label">建议继续追问</span>
                    <ChipSet
                      items={result.possible_additional_symptoms}
                      tone="gold"
                      emptyText="暂无补充追问项"
                    />
                  </div>

                  <div className="info-panel">
                    <span className="panel-label">图谱检索摘要</span>
                    <p className="supporting-copy strong-copy">
                      {result.kg_context?.retrieval_summary || '暂无图谱检索摘要。'}
                    </p>
                  </div>

                  <div className="info-panel full-span">
                    <span className="panel-label">证据路径</span>
                    {result.kg_context?.graph_paths?.length ? (
                      <div className="path-stack">
                        {result.kg_context.graph_paths.map((item) => (
                          <code key={item} className="path-card">
                            {item}
                          </code>
                        ))}
                      </div>
                    ) : (
                      <span className="empty-copy">暂无图谱路径</span>
                    )}
                  </div>
                </div>
              </Section>

              <Section eyebrow="处置建议" title="检查、治疗与日常照护">
                <div className="report-grid report-grid-two">
                  <div className="info-panel">
                    <span className="panel-label">推荐检查</span>
                    <ChipSet items={result.recommended_tests} tone="rust" emptyText="暂无推荐检查" />
                  </div>

                  <div className="info-panel">
                    <span className="panel-label">治疗方式</span>
                    <ChipSet
                      items={result.treatment_plan?.therapies}
                      tone="slate"
                      emptyText="暂无明确治疗方式"
                    />
                  </div>

                  <div className="info-panel">
                    <span className="panel-label">可能涉及的药物</span>
                    <ChipSet
                      items={result.treatment_plan?.medications}
                      tone="neutral"
                      emptyText="暂无明确药物线索"
                    />
                  </div>

                  <div className="info-panel">
                    <span className="panel-label">照护提醒</span>
                    <ReportList
                      items={result.treatment_plan?.care_points}
                      emptyText="暂无额外照护提醒"
                    />
                  </div>

                  <div className="info-panel">
                    <span className="panel-label">建议饮食</span>
                    <ChipSet
                      items={result.treatment_plan?.diet_recommendations}
                      tone="teal"
                      emptyText="暂无明确饮食建议"
                    />
                  </div>

                  <div className="info-panel">
                    <span className="panel-label">饮食禁忌</span>
                    <ChipSet
                      items={result.treatment_plan?.diet_avoid}
                      tone="rust"
                      emptyText="暂无明显禁忌"
                    />
                  </div>
                </div>
              </Section>

              {result.warning ? (
                <Alert
                  className="report-alert"
                  type="warning"
                  showIcon
                  message="风险提醒"
                  description={result.warning}
                />
              ) : null}

              <Alert
                className="report-alert"
                type="info"
                showIcon
                icon={<SafetyCertificateOutlined />}
                message="医学声明"
                description={result.disclaimer}
              />
            </div>
          )}
        </section>
      </main>
    </div>
  );
};

export default App;
