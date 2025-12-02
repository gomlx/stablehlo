package utils

import (
	"testing"
)

func TestSet(t *testing.T) {
	// Sets are created empty.
	s := MakeSet[int](10)
	if len(s) != 0 {
		t.Errorf("expected len 0, got %d", len(s))
	}

	// Check inserting and recovery.
	s.Insert(3, 7)
	if len(s) != 2 {
		t.Errorf("expected len 2, got %d", len(s))
	}
	if !s.Has(3) {
		t.Errorf("expected s.Has(3) to be true")
	}
	if !s.Has(7) {
		t.Errorf("expected s.Has(7) to be true")
	}
	if s.Has(5) {
		t.Errorf("expected s.Has(5) to be false")
	}

	s2 := SetWith(5, 7)
	if len(s2) != 2 {
		t.Errorf("expected len 2, got %d", len(s2))
	}
	if !s2.Has(5) {
		t.Errorf("expected s2.Has(5) to be true")
	}
	if !s2.Has(7) {
		t.Errorf("expected s2.Has(7) to be true")
	}
	if s2.Has(3) {
		t.Errorf("expected s2.Has(3) to be false")
	}

	s3 := s.Sub(s2)
	if len(s3) != 1 {
		t.Errorf("expected len 1, got %d", len(s3))
	}
	if !s3.Has(3) {
		t.Errorf("expected s3.Has(3) to be true")
	}

	delete(s, 7)
	if len(s) != 1 {
		t.Errorf("expected len 1, got %d", len(s))
	}
	if !s.Has(3) {
		t.Errorf("expected s.Has(3) to be true")
	}
	if s.Has(7) {
		t.Errorf("expected s.Has(7) to be false")
	}
	if !s.Equal(s3) {
		t.Errorf("expected s.Equal(s3) to be true")
	}
	if s.Equal(s2) {
		t.Errorf("expected s.Equal(s2) to be false")
	}
	s4 := SetWith(-3)
	if s.Equal(s4) {
		t.Errorf("expected s.Equal(s4) to be false")
	}
}
