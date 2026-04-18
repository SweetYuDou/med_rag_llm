import React from 'react';
import { Empty, Typography } from 'antd';

import PillList from './PillList.jsx';

const { Paragraph, Text } = Typography;

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

const AssistantReport = ({ report }) => {
  if (!report) {
    return null;
  }

  const triageTone = triageToneMap[report.triage_level] || 'calm';

  return (
    <div className="assistant-report">
      <div className="report-summary-card">
        <div className="report-summary-head">
          <span className={`triage-pill triage-pill-${triageTone}`}>{report.triage_level}</span>
          <span className="report-meta-chip">{report.recommended_department?.primary || '全科医学科'}</span>
        </div>
        <Paragraph className="report-advice">{report.advice}</Paragraph>
        <div className="report-context-row">
          <span className="report-mini-chip">
            病程：
            {durationOptions.find((item) => item.value === report.context?.duration)?.label || '未填写'}
          </span>
          <span className="report-mini-chip">
            严重程度：
            {severityOptions.find((item) => item.value === report.context?.severity)?.label || '未填写'}
          </span>
        </div>
        <Text type="secondary">{report.retrieval_summary}</Text>
      </div>

      <div className="report-grid">
        <section className="report-card">
          <div className="report-card-title">已识别症状</div>
          <PillList items={report.recognized_symptoms} emptyText="还没有稳定命中的症状节点" tone="cyan" />
        </section>

        <section className="report-card">
          <div className="report-card-title">推荐检查</div>
          <PillList items={report.recommended_tests} emptyText="暂无推荐检查" tone="blue" />
        </section>

        <section className="report-card report-card-wide">
          <div className="report-card-title">候选疾病</div>
          {report.top_diseases?.length ? (
            <div className="disease-grid">
              {report.top_diseases.map((disease) => (
                <article key={disease.name} className="disease-card">
                  <div className="disease-card-head">
                    <strong>{disease.name}</strong>
                    <span className="disease-probability">{disease.probability}</span>
                  </div>
                  <div className="disease-card-block">
                    <Text strong>匹配症状</Text>
                    <PillList items={disease.matched_symptoms} emptyText="暂无" tone="cyan" />
                  </div>
                  <div className="disease-card-block">
                    <Text strong>相关科室</Text>
                    <PillList items={disease.recommended_department} emptyText="全科医学科" tone="green" />
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无稳定的候选疾病" />
          )}
        </section>

        <section className="report-card">
          <div className="report-card-title">治疗与照护</div>
          <PillList
            items={report.treatment_plan?.medications}
            emptyText="暂无明确治疗线索"
            tone="amber"
          />
          <div className="report-card-block">
            <Text strong>照护提醒</Text>
            <div className="bullet-list">
              {(report.treatment_plan?.care_points || []).map((item) => (
                <div key={item} className="bullet-item">
                  {item}
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="report-card">
          <div className="report-card-title">建议继续追问</div>
          <div className="bullet-list">
            {(report.follow_up_questions || []).map((item) => (
              <div key={item} className="bullet-item">
                {item}
              </div>
            ))}
            {!report.follow_up_questions?.length && <Text type="secondary">当前没有额外追问项。</Text>}
          </div>
        </section>

        <section className="report-card report-card-wide">
          <div className="report-card-title">图谱证据</div>
          <div className="citation-list">
            {(report.citations || []).map((citation) => (
              <article key={`${citation.title}-${citation.snippet}`} className="citation-card">
                <strong>{citation.title}</strong>
                <span>{citation.snippet}</span>
                <em>{citation.source}</em>
              </article>
            ))}
            {!report.citations?.length && <Text type="secondary">暂无可展示的检索证据。</Text>}
          </div>
        </section>
      </div>
    </div>
  );
};

export default AssistantReport;
