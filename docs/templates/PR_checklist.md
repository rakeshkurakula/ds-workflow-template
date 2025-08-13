# Pull Request Checklist

## Pre-submission Requirements
*Complete all applicable items before submitting your PR*

### Code Quality
- [ ] **Tests added/updated**: New functionality has corresponding tests
- [ ] **Tests passing**: All existing tests continue to pass
- [ ] **Code reviewed**: Code follows team standards and best practices
- [ ] **Documentation updated**: README, docstrings, or other docs reflect changes
- [ ] **Linting passed**: Code passes all linting and formatting checks

### Data Validation
- [ ] **Data validation passing**: All data quality checks are green
- [ ] **Data schema validated**: New data follows expected schema
- [ ] **Data pipeline tested**: End-to-end data flow has been verified
- [ ] **Data drift checked**: Monitored for unexpected changes in data distribution

### Model and Experiment Tracking
- [ ] **MLflow run linked**: PR includes link to MLflow experiment run
- [ ] **Dataset hash included**: Unique identifier for training/validation data
- [ ] **Git SHA recorded**: Commit hash is tracked in experiment metadata
- [ ] **Model artifacts logged**: All model files and dependencies are versioned
- [ ] **Hyperparameters documented**: All model settings are recorded

### Documentation and Communication
- [ ] **Model card updated**: Model documentation reflects current state
- [ ] **Experiment results documented**: Key findings and decisions are recorded
- [ ] **Performance metrics logged**: Baseline and new model metrics compared
- [ ] **Business impact assessed**: Performance change impact on KPIs evaluated

### Deployment Readiness
- [ ] **Rollback plan noted**: Clear steps for reverting changes if needed
- [ ] **Monitoring plan updated**: Alerts and dashboards account for changes
- [ ] **Infrastructure requirements**: Resource needs documented and approved
- [ ] **A/B testing plan**: Strategy for gradual rollout documented (if applicable)

### Security and Compliance
- [ ] **Security review**: No sensitive data or credentials in code
- [ ] **Data privacy assessed**: Changes comply with data protection policies
- [ ] **Access controls verified**: Appropriate permissions are in place

## PR Description Template
*Use this template for your PR description*

```markdown
## Summary
Brief description of changes made

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing performed

## MLflow Run
- **Run ID**: [link_to_mlflow_run]
- **Dataset Hash**: [dataset_hash]
- **Git SHA**: [commit_hash]

## Performance Impact
- **Baseline Metric**: X.XX
- **New Metric**: X.XX
- **Change**: +/- X.XX%

## Rollback Plan
- Step 1: [specific_action]
- Step 2: [specific_action]
- Step 3: [specific_action]

## Additional Notes
[Any additional context or considerations]
```

## Reviewer Guidelines
*For reviewers of this PR*

### Technical Review
- [ ] Code logic is sound and efficient
- [ ] Error handling is appropriate
- [ ] Code is well-documented and maintainable
- [ ] No obvious security vulnerabilities

### Data Science Review
- [ ] Model approach is justified
- [ ] Evaluation methodology is appropriate
- [ ] Results are statistically significant
- [ ] Model interpretability considerations addressed

### Business Review
- [ ] Solution aligns with business requirements
- [ ] Performance improvement is meaningful
- [ ] Implementation timeline is reasonable
- [ ] Resource requirements are acceptable

---

**Remember**: All items should be completed before merging. If any item is not applicable, mark it as N/A in the PR description.
