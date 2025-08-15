/* Copyright 2025 Daniil Shmelev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ========================================================================= */

#pragma once
#include "cppch.h"

#pragma pack(push)
//#pragma pack(1)

template<typename T>
class PointImpl;

template<typename T>
class PointImplTimeAug;

template<typename T>
class PointImplLeadLag;

template<typename T>
class PointImplTimeAugLeadLag;

template<typename T>
class Point;

template<typename T>
class Path {
public:
	static_assert(std::is_arithmetic<T>::value);

	Path(T* data_, uint64_t dimension_, uint64_t length_, bool time_aug_ = false, bool lead_lag_ = false, double end_time_ = 1.) :
		_dimension{ (lead_lag_ ? 2 * dimension_ : dimension_) + (time_aug_ ? 1 : 0) },
		_length{ lead_lag_ ? length_ * 2 - 1 : length_ },
		_data{ std::span<T>(data_, dimension_ * length_) },
		_data_dimension{ dimension_ },
		_data_length{ length_ },
		_data_size{ dimension_ * length_ },
		_time_aug{ time_aug_ },
		_lead_lag{ lead_lag_ },
		_time_step{ end_time_ / (_length - 1) } {
	}

	Path(const std::span<T> data_, uint64_t dimension_, uint64_t length_, bool time_aug_ = false, bool lead_lag_ = false, double end_time_ = 1.) :
		_dimension{ (lead_lag_ ? 2 * dimension_ : dimension_) + (time_aug_ ? 1 : 0) },
		_length{ lead_lag_ ? length_ * 2 - 1 : length_ },
		_data{ data_ },
		_data_dimension{ dimension_ },
		_data_length{ length_ },
		_data_size{ dimension_ * length_ },
		_time_aug{ time_aug_ },
		_lead_lag{ lead_lag_ },
		_time_step{ end_time_ / (_length - 1) } {
		if (data_.size() != dimension_ * length_)
			throw std::invalid_argument("1D vector is not the correct shape for a path of dimension " + std::to_string(dimension_) + " and length " + std::to_string(length_));
	}

	Path(const Path& other) :
		_dimension{ other._dimension },
		_length{ other._length },
		_data{ other._data },
		_data_dimension{ other._data_dimension },
		_data_length{ other._data_length },
		_data_size{ other._data_size },
		_time_aug{ other._time_aug },
		_lead_lag{ other._lead_lag },
		_time_step{ other._time_step } {
	}

	Path(const Path& other, bool time_aug_, bool lead_lag_, double end_time_ = 1.) :
		_dimension{ (lead_lag_ ? 2 * other._data_dimension : other._data_dimension) + (time_aug_ ? 1 : 0) },
		_length{ lead_lag_ ? other._data_length * 2 - 1 : other._data_length },
		_data{ other._data },
		_data_dimension{ other._data_dimension },
		_data_length{ other._data_length },
		_data_size{ other._data_size },
		_time_aug{ time_aug_ },
		_lead_lag{ lead_lag_ },
		_time_step{ end_time_ / (_length - 1) } {
	}

	virtual ~Path() {}

	Path<T>& operator=(const Path&) = delete;

	inline uint64_t dimension() const { return _dimension; }
	inline uint64_t data_dimension() const { return _data_dimension; }
	inline uint64_t length() const { return _length; }
	inline uint64_t data_length() const { return _data_length; }
	inline T* data() const { return _data.data(); }

	inline bool time_aug() const { return _time_aug; }
	inline bool lead_lag() const { return _lead_lag; }
	inline double time_step() const { return _time_step; }
	inline double end_time() const { return _time_step * (_length - 1); }

	friend class Point<T>;
	friend class PointImpl<T>;
	friend class PointImplTimeAug<T>;
	friend class PointImplLeadLag<T>;
	friend class PointImplTimeAugLeadLag<T>;

	Point<T> operator[](uint64_t i) const { 
#ifdef _DEBUG
		if (i < 0 || i >= _length)
			throw std::out_of_range("Argument out of bounds in Path::operator[]");
#endif
		return Point<T>(this, i);
	}

	inline Point<T> begin() const
	{
		return Point<T>(this, 0);
	}
	inline Point<T> end() const
	{
		return Point<T>(this, _length); 
	}

	bool operator==(const Path& other) const {
		return _data.data() == other._data.data()
			&& _time_aug == other._time_aug
			&& _lead_lag == other._lead_lag;
	}
	bool operator!=(const Path& other) const {
		return !this->operator==(other);
	}

	PointImpl<T>* point_impl_factory(uint64_t index) const;

private:
	const uint64_t _dimension;
	const uint64_t _length;

	const std::span<T> _data;
	const uint64_t _data_dimension;
	const uint64_t _data_length;
	const uint64_t _data_size;

	const bool _time_aug;
	const bool _lead_lag;
	const double _time_step;
};

template<typename T>
class PointImpl {
public:
	PointImpl() : ptr{ nullptr }, path{ nullptr } {}
	PointImpl(const Path<T>* path_, uint64_t index) :
		ptr{ path_->_data.data() + index * path_->_data_dimension },
		path{ path_ }
	{}
	PointImpl(const PointImpl& other) : 
		ptr{ other.ptr },
		path{ other.path }
	{}
	virtual ~PointImpl() {}

	virtual PointImpl<T>* duplicate() const {
		auto p = new PointImpl();
		p->ptr = ptr;
		p->path = path;
		return p;
	}

	virtual inline double operator[](uint64_t i) const { return static_cast<double>(ptr[i]); }
	virtual inline void operator++() { ptr += path->_data_dimension; }
	virtual inline void operator--() { ptr -= path->_data_dimension; }

	inline uint64_t dimension() { return path->_dimension; }
	virtual inline void advance(int64_t n) { ptr += n * path->_data_dimension; }
	virtual inline void set_to_start() { ptr = path->_data.data(); }
	virtual inline void set_to_end() { ptr = path->_data.data() + path->_data_size; }
	virtual inline void set_to_index(int64_t n) { ptr = path->_data.data() + n * path->_data_dimension; }

	inline T* data() const { return ptr; }
	virtual inline uint64_t index() const { return static_cast<uint64_t>((ptr - path->_data.data()) / path->_data_dimension); }

	// We assume here that the two points being compared belong to the same path,
	// which saves us checking this each time. We keep this check in debug.

#ifndef _DEBUG
	virtual bool operator==(const PointImpl& other) const { return ptr == other.ptr; }
	virtual bool operator<(const PointImpl& other) const { return ptr < other.ptr; }
	virtual bool operator>(const PointImpl& other) const { return ptr > other.ptr; }
#else
	virtual bool operator==(const PointImpl& other) const { return path == other.path && ptr == other.ptr; }
	virtual bool operator<(const PointImpl& other) const { return path == other.path && ptr < other.ptr; }
	virtual bool operator>(const PointImpl& other) const { return path == other.path && ptr > other.ptr; }
#endif
	bool operator!=(const PointImpl& other) const { return !(*this == other); }
	bool operator<=(const PointImpl& other) const { return !(*this > other); }
	bool operator>=(const PointImpl& other) const { return !(*this < other); }

	T* ptr;
	const Path<T>* path;
};

template<typename T>
class PointImplTimeAug : public PointImpl<T> {
public:
	PointImplTimeAug() : PointImpl<T>(), time{ 0. } {}
	PointImplTimeAug(const Path<T>* path_, uint64_t index) : PointImpl<T>(path_, index), time{ index * this->path->_time_step } {}
	PointImplTimeAug(const PointImplTimeAug& other) : PointImpl<T>(other), time{ other.time } {}
	virtual ~PointImplTimeAug() {}

	PointImpl<T>* duplicate() const override {
		auto p = new PointImplTimeAug();
		p->ptr = this->ptr;
		p->path = this->path;
		p->time = this->time;
		return p;
	}

	inline double operator[](uint64_t i) const override { return static_cast<double>((i < this->path->_data_dimension) ? this->ptr[i] : time);	}
	inline void operator++() override { this->ptr += this->path->_data_dimension; time += this->path->_time_step;	}
	inline void operator--() override { this->ptr -= this->path->_data_dimension; time -= this->path->_time_step; }
	inline void advance(int64_t n) override { this->ptr += n * this->path->_data_dimension; time += n * this->path->_time_step; }
	inline void set_to_start() override { this->ptr = this->path->_data.data(); time = 0.; }
	inline void set_to_end() override { this->ptr = this->path->_data.data() + this->path->_data_size; time = this->path->_length * this->path->_time_step; }
	inline void set_to_index(int64_t n) override { this->ptr = this->path->_data.data() + n * this->path->_data_dimension; time = n * this->path->_time_step; }

private:
	double time;
};

template<typename T>
class PointImplLeadLag : public PointImpl<T> {
public:
	PointImplLeadLag() : PointImpl<T>(), parity{ false } {}
	PointImplLeadLag(const Path<T>* path_, uint64_t index) : PointImpl<T>(path_, index / 2), parity{ static_cast<bool>(index % 2) } {}
	PointImplLeadLag(const PointImplLeadLag& other) : PointImpl<T>(other), parity{ other.parity } {}
	virtual ~PointImplLeadLag() {}

	PointImpl<T>* duplicate() const override {
		auto p = new PointImplLeadLag();
		p->ptr = this->ptr;
		p->path = this->path;
		p->parity = this->parity;
		return p;
	}

	inline double operator[](uint64_t i) const override { 
		if (i < this->path->_data_dimension)
			return static_cast<double>(this->ptr[i]);
		else {
			uint64_t leadIdx = parity ? i : i - this->path->_data_dimension;
			return static_cast<double>(this->ptr[leadIdx]);
		}
	}
	inline void operator++() override { if (parity) this->ptr += this->path->_data_dimension; parity = !parity; }
	inline void operator--() override { if (!parity) this->ptr -= this->path->_data_dimension; parity = !parity; }
	inline void advance(int64_t n) override { this->ptr += (n / 2) * this->path->_data_dimension; parity = (parity != static_cast<bool>(n % 2)); }
	inline void set_to_start() override { this->ptr = this->path->_data.data(); parity = false; }
	inline void set_to_end() override { this->ptr = this->path->_data.data() + this->path->_data_size; parity = true; }
	inline void set_to_index(int64_t n) override { this->ptr = this->path->_data.data() + (n / 2) * this->path->_data_dimension; parity = static_cast<bool>(n % 2); }

	inline uint64_t index() const override { return 2 * static_cast<uint64_t>(this->ptr - this->path->_data.data()) + static_cast<uint64_t>(parity); }

	bool operator==(const PointImpl<T>& other) const override {
		if (!PointImpl<T>::operator==(other))
			return false;
		return parity == static_cast<const PointImplLeadLag<T>*>(&other)->parity;
	}
	bool operator<(const PointImpl<T>& other) const override {
		if (PointImpl<T>::operator<(other))
			return true;
		return PointImpl<T>::operator==(other) && parity < static_cast<const PointImplLeadLag<T>*>(&other)->parity;
	}
	bool operator>(const PointImpl<T>& other) const override {
		if (PointImpl<T>::operator>(other))
			return true;
		return PointImpl<T>::operator==(other) && parity > static_cast<const PointImplLeadLag<T>*>(&other)->parity;
	}

private:
	bool parity;
};

template<typename T>
class PointImplTimeAugLeadLag : public PointImpl<T> {
public:
	PointImplTimeAugLeadLag() : PointImpl<T>(), parity{ false }, time{ 0. }, _data_dimension_times_2{ 0 } {}
	PointImplTimeAugLeadLag(const Path<T>* path_, uint64_t index) : 
		PointImpl<T>(path_, index / 2), 
		parity{ static_cast<bool>(index % 2) },
		time{ index * this->path->_time_step },
		_data_dimension_times_2{ path_->_data_dimension * 2 }
	{}
	PointImplTimeAugLeadLag(const PointImplTimeAugLeadLag& other) :
		PointImpl<T>(other), 
		parity{ other.parity },
		time{ other.time },
		_data_dimension_times_2{ other._data_dimension_times_2 }{}
	virtual ~PointImplTimeAugLeadLag() {}

	PointImpl<T>* duplicate() const override {
		auto p = new PointImplTimeAugLeadLag();
		p->ptr = this->ptr;
		p->path = this->path;
		p->parity = this->parity;
		p->time = this->time;
		p->_data_dimension_times_2 = this->_data_dimension_times_2;
		return p;
	}

	inline double operator[](uint64_t i) const override {
		if (i < this->path->_data_dimension)
			return static_cast<double>(this->ptr[i]);
		else if (i < this->_data_dimension_times_2) {
			uint64_t lead_idx = parity ? i : i - this->path->_data_dimension;
			return static_cast<double>(this->ptr[lead_idx]);
		}
		else
			return time;
	}
	inline void operator++() override {
		if (parity) { this->ptr += this->path->_data_dimension; }
		time += this->path->_time_step;
		parity = !parity;
	}
	inline void operator--() override {
		if (!parity) { this->ptr -= this->path->_data_dimension; }
		time -= this->path->_time_step;
		parity = !parity;
	}
	inline void advance(int64_t n) override { 
		this->ptr += (n / 2) * this->path->_data_dimension; 
		parity = (parity != static_cast<bool>(n % 2)); 
		time += n * this->path->_time_step;
	}
	inline void set_to_start() override { this->ptr = this->path->_data.data(); parity = false; time = 0; }
	inline void set_to_end() override { this->ptr = this->path->_data.data() + this->path->_data_size; parity = true; time = this->path->_length * this->path->_time_step; }
	inline void set_to_index(int64_t n) override { 
		this->ptr = this->path->_data.data() + (n / 2) * this->path->_data_dimension;
		parity = static_cast<bool>(n % 2);
		time = n * this->path->_time_step;
	}

	inline uint64_t index() const override { return 2UL * static_cast<uint64_t>(this->ptr - this->path->_data.data()) + static_cast<uint64_t>(parity); }

	bool operator==(const PointImpl<T>& other) const override {
		if (!PointImpl<T>::operator==(other))
			return false;
		return parity == static_cast<const PointImplTimeAugLeadLag<T>*>(&other)->parity;
	}
	bool operator<(const PointImpl<T>& other) const override {
		if (PointImpl<T>::operator<(other))
			return true;
		return PointImpl<T>::operator==(other) && parity < static_cast<const PointImplTimeAugLeadLag<T>*>(&other)->parity;
	}
	bool operator>(const PointImpl<T>& other) const override {
		if (PointImpl<T>::operator>(other))
			return true;
		return PointImpl<T>::operator==(other) && parity > static_cast<const PointImplTimeAugLeadLag<T>*>(&other)->parity;
	}

private:
	bool parity;
	double time;
	uint64_t _data_dimension_times_2;
};

template<typename T>
class Point {
public:
	static_assert(std::is_arithmetic<T>::value);

	Point() {
		_impl.reset(nullptr);
	}

	Point(const Path<T>* path, uint64_t index) {
		// Create new impl from path and index - path knows how
		_impl.reset(path->point_impl_factory(index));
	}

	Point(const Point& other) {
		_impl.reset(other._impl->duplicate());
	}

	Point(Point&& other)
	{
		_impl.swap(other._impl);
	}

	Point& operator=(const Point& other) {
		if (this != &other) {
			_impl.reset(other._impl->duplicate());
		}
		return *this;
	}

	Point& operator=(Point&& other) {
		_impl.swap(other._impl);
		return *this;
	}


	inline double operator[](uint64_t i) const { 
#ifdef _DEBUG
		sq_bracket_bounds_check(i);
#endif
		return _impl->operator[](i);
	}
	inline Point& operator++() {
		_impl->operator++();
		return *this;
	}
	inline Point operator++(int) {
		Point tmp{ *this };
		++(*this);
		return tmp;
	}
	inline Point& operator--() {
		_impl->operator--();
		return *this;
	}
	inline Point operator--(int) {
		Point tmp{ *this };
		--(*this);
		return tmp;
	}

	inline uint64_t dimension() { return _impl->dimension(); }
	inline void advance(int64_t n) { 
		_impl->advance(n);
	}
	inline void set_to_start() { _impl->set_to_start(); }
	inline void set_to_end() { _impl->set_to_end(); }
	inline void set_to_index(int64_t n) { _impl->set_to_index(); }

	inline T* data() const { return _impl->data(); }
	inline uint64_t index() const { 
#ifdef _DEBUG
		index_bounds_check();
#endif
		return _impl->index();
	}

	bool operator==(const Point& other) const { return _impl->operator==(*other._impl); }
	bool operator!=(const Point& other) const { return _impl->operator!=(*other._impl); }
	bool operator<(const Point& other) const { return _impl->operator<(*other._impl); }
	bool operator<=(const Point& other) const { return _impl->operator<=(*other._impl); }
	bool operator>(const Point& other) const { return _impl->operator>(*other._impl); }
	bool operator>=(const Point& other) const { return _impl->operator>=(*other._impl); }
private:
	std::unique_ptr<PointImpl<T>> _impl;

#ifdef _DEBUG
	inline void sq_bracket_bounds_check(uint64_t i) const {
		if (_impl->ptr < _impl->path->_data.data() || _impl->ptr >= _impl->path->_data.data() + _impl->path->_data_size)
			throw std::out_of_range("Point is out of bounds for given path in Point::operator[]");

		if (i < 0 || i >= _impl->path->_dimension)
			throw std::out_of_range("Argument out of bounds in Point::operator[]");
	}

	inline void index_bounds_check() const {
		if (_impl->ptr < _impl->path->_data.data() || _impl->ptr >= _impl->path->_data.data() + _impl->path->_data_size)
			throw std::out_of_range("Point is out of bounds for given path in Point::index()");
	}
#endif
};

#pragma pack(pop)
