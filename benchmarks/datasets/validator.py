"""
Dataset validation utilities.

Checks dataset quality, consistency, and completeness.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from benchmarks.schemas import (
    DS1Dataset, DS2Dataset, DS3Dataset, DS4Dataset, DS5Dataset
)


class ValidationIssue(BaseModel):
    """A validation issue found in a dataset."""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'missing_field', 'invalid_value', 'inconsistency', etc.
    message: str
    example_id: Optional[str] = None
    field: Optional[str] = None


class ValidationReport(BaseModel):
    """Report of validation results."""
    dataset_name: str
    dataset_type: str
    total_examples: int
    
    errors: List[ValidationIssue] = Field(default_factory=list)
    warnings: List[ValidationIssue] = Field(default_factory=list)
    info: List[ValidationIssue] = Field(default_factory=list)
    
    is_valid: bool = True
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue."""
        if issue.severity == 'error':
            self.errors.append(issue)
            self.is_valid = False
        elif issue.severity == 'warning':
            self.warnings.append(issue)
        else:
            self.info.append(issue)
    
    def get_summary(self) -> str:
        """Get summary string."""
        return (
            f"Validation Report: {self.dataset_name}\n"
            f"  Total examples: {self.total_examples}\n"
            f"  Errors: {len(self.errors)}\n"
            f"  Warnings: {len(self.warnings)}\n"
            f"  Info: {len(self.info)}\n"
            f"  Valid: {self.is_valid}"
        )


class DatasetValidator:
    """
    Validator for benchmark datasets.
    
    Checks for common issues and quality problems.
    """
    
    def __init__(self, strict: bool = False):
        """
        Initialize validator.
        
        Args:
            strict: If True, treat warnings as errors
        """
        self.strict = strict
    
    def validate_ds1(self, dataset: DS1Dataset) -> ValidationReport:
        """
        Validate DS1 dataset.
        
        Checks:
        - All examples have IDs
        - All questions are non-empty
        - All answers have citations
        - No duplicate IDs
        """
        report = ValidationReport(
            dataset_name=dataset.name,
            dataset_type="DS1",
            total_examples=len(dataset.examples)
        )
        
        seen_ids = set()
        
        for ex in dataset.examples:
            # Check ID
            if not ex.id:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='missing_field',
                    message='Example missing ID',
                    example_id=ex.id
                ))
            elif ex.id in seen_ids:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='duplicate',
                    message=f'Duplicate ID: {ex.id}',
                    example_id=ex.id
                ))
            else:
                seen_ids.add(ex.id)
            
            # Check question
            if not ex.question or len(ex.question.strip()) < 10:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='invalid_value',
                    message='Question too short or empty',
                    example_id=ex.id,
                    field='question'
                ))
            
            # Check answer
            if not ex.gold_answer or len(ex.gold_answer.strip()) < 20:
                report.add_issue(ValidationIssue(
                    severity='warning',
                    category='invalid_value',
                    message='Answer suspiciously short',
                    example_id=ex.id,
                    field='gold_answer'
                ))
            
            # Check citations
            if not ex.gold_citations:
                report.add_issue(ValidationIssue(
                    severity='warning' if not self.strict else 'error',
                    category='missing_field',
                    message='No citations provided',
                    example_id=ex.id,
                    field='gold_citations'
                ))
            
            # Check topic
            if not ex.topic:
                report.add_issue(ValidationIssue(
                    severity='info',
                    category='missing_field',
                    message='No topic specified',
                    example_id=ex.id,
                    field='topic'
                ))
        
        return report
    
    def validate_ds2(self, dataset: DS2Dataset) -> ValidationReport:
        """Validate DS2 dataset."""
        report = ValidationReport(
            dataset_name=dataset.name,
            dataset_type="DS2",
            total_examples=len(dataset.examples)
        )
        
        seen_ids = set()
        
        for ex in dataset.examples:
            # Check ID
            if not ex.id:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='missing_field',
                    message='Example missing ID',
                    example_id=ex.id
                ))
            elif ex.id in seen_ids:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='duplicate',
                    message=f'Duplicate ID: {ex.id}',
                    example_id=ex.id
                ))
            else:
                seen_ids.add(ex.id)
            
            # Check query
            if not ex.query or len(ex.query.strip()) < 5:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='invalid_value',
                    message='Query too short or empty',
                    example_id=ex.id,
                    field='query'
                ))
            
            # Check qrels
            if not ex.qrels:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='missing_field',
                    message='No qrels provided',
                    example_id=ex.id,
                    field='qrels'
                ))
            else:
                # Check for at least one relevant item
                relevant_count = sum(1 for q in ex.qrels if q.relevance > 0)
                if relevant_count == 0:
                    report.add_issue(ValidationIssue(
                        severity='warning',
                        category='invalid_value',
                        message='No relevant items in qrels',
                        example_id=ex.id,
                        field='qrels'
                    ))
        
        return report
    
    def validate_ds3(self, dataset: DS3Dataset) -> ValidationReport:
        """Validate DS3 dataset."""
        report = ValidationReport(
            dataset_name=dataset.name,
            dataset_type="DS3",
            total_examples=len(dataset.examples)
        )
        
        seen_ids = set()
        
        for ex in dataset.examples:
            # Check ID
            if not ex.id:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='missing_field',
                    message='Example missing ID',
                    example_id=ex.id
                ))
            elif ex.id in seen_ids:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='duplicate',
                    message=f'Duplicate ID: {ex.id}',
                    example_id=ex.id
                ))
            else:
                seen_ids.add(ex.id)
            
            # Check question
            if not ex.question:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='missing_field',
                    message='Question missing',
                    example_id=ex.id,
                    field='question'
                ))
            
            # Check reason
            if not ex.reason_unanswerable:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='missing_field',
                    message='Reason for unanswerability missing',
                    example_id=ex.id,
                    field='reason_unanswerable'
                ))
        
        return report
    
    def validate_ds4(self, dataset: DS4Dataset) -> ValidationReport:
        """Validate DS4 dataset."""
        report = ValidationReport(
            dataset_name=dataset.name,
            dataset_type="DS4",
            total_examples=len(dataset.examples)
        )
        
        seen_ids = set()
        
        for ex in dataset.examples:
            # Check ID uniqueness
            if ex.id in seen_ids:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='duplicate',
                    message=f'Duplicate ID: {ex.id}',
                    example_id=ex.id
                ))
            else:
                seen_ids.add(ex.id)
            
            # Check hops
            if not ex.hops or len(ex.hops) < 2:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='invalid_value',
                    message='Multi-hop question must have at least 2 hops',
                    example_id=ex.id,
                    field='hops'
                ))
            
            # Check citations
            if not ex.gold_citations:
                report.add_issue(ValidationIssue(
                    severity='warning',
                    category='missing_field',
                    message='No citations provided',
                    example_id=ex.id,
                    field='gold_citations'
                ))
        
        return report
    
    def validate_ds5(self, dataset: DS5Dataset) -> ValidationReport:
        """Validate DS5 dataset."""
        report = ValidationReport(
            dataset_name=dataset.name,
            dataset_type="DS5",
            total_examples=len(dataset.examples)
        )
        
        seen_ids = set()
        
        for ex in dataset.examples:
            # Check ID uniqueness
            if ex.id in seen_ids:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='duplicate',
                    message=f'Duplicate ID: {ex.id}',
                    example_id=ex.id
                ))
            else:
                seen_ids.add(ex.id)
            
            # Check structured output
            if not ex.structured_output.gold_output:
                report.add_issue(ValidationIssue(
                    severity='error',
                    category='missing_field',
                    message='Gold output missing',
                    example_id=ex.id,
                    field='structured_output.gold_output'
                ))
            
            # Check required fields
            if not ex.required_fields:
                report.add_issue(ValidationIssue(
                    severity='warning',
                    category='missing_field',
                    message='No required fields specified',
                    example_id=ex.id,
                    field='required_fields'
                ))
        
        return report
    
    def validate_any(self, dataset) -> ValidationReport:
        """Validate any dataset type."""
        if isinstance(dataset, DS1Dataset):
            return self.validate_ds1(dataset)
        elif isinstance(dataset, DS2Dataset):
            return self.validate_ds2(dataset)
        elif isinstance(dataset, DS3Dataset):
            return self.validate_ds3(dataset)
        elif isinstance(dataset, DS4Dataset):
            return self.validate_ds4(dataset)
        elif isinstance(dataset, DS5Dataset):
            return self.validate_ds5(dataset)
        else:
            raise ValueError(f"Unknown dataset type: {type(dataset)}")

